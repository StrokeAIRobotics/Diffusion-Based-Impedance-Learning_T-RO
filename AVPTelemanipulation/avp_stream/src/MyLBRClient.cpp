/**
DISCLAIMER OF WARRANTY
...
\version {1.9}
*/
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

using namespace std;
namespace py = pybind11;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#ifndef NCoef
#define NCoef 1
#endif

static double filterOutput[7][NCoef + 1];
static double filterInput[7][NCoef + 1];


/**
* \brief Initialization
*
*/
MyLBRClient::MyLBRClient(double freqHz, double amplitude)
    :guard{} // Initialize guard (Python interpreter)
{

    /** Initialization */
    // TODO[CONFIG]: Verify/replace with YOUR initial robot configuration (must match Java app)
    qInitial[0] = 5.59 * M_PI/180;
    qInitial[1] = 50.76 * M_PI/180;
    qInitial[2] = 0.04 * M_PI/180;
    qInitial[3] = -86.34 * M_PI/180;
    qInitial[4] = -3.49 * M_PI/180;
    qInitial[5] = 40.25 * M_PI/180;
    qInitial[6] = 92.61 * M_PI/180;

    // Create robot model
    myLBR = new iiwa14( 1, "Trey");
    myLBR->init( );

    // Current joint configuration and velocity
    q  = Eigen::VectorXd::Zero( myLBR->nq );
    q_ini = Eigen::VectorXd::Zero( myLBR->nq );
    dq = Eigen::VectorXd::Zero( myLBR->nq );

    // Time variables for control loop
    currentTime = 0;
    sampleTime = 0;

    // Initialize joint torques and joint positions (also needed for waitForCommand()!)
    for( int i=0; i < myLBR->nq; i++ )
    {
        qCurr[i] = qInitial[i];
        qOld[i] = qInitial[i];
        qApplied[i] = 0.0;
        torques[i] = 0.0;
    }

    tau_motion    = Eigen::VectorXd::Zero( myLBR->nq );
    tau_previous  = Eigen::VectorXd::Zero( myLBR->nq );
    tau_prev_prev = Eigen::VectorXd::Zero( myLBR->nq );
    tau_total     = Eigen::VectorXd::Zero( myLBR->nq );

    // ************************************************************
    // INITIALIZE YOUR VECTORS AND MATRICES HERE
    // ************************************************************
    M = Eigen::MatrixXd::Zero( 7, 7 );
    M_inv = Eigen::MatrixXd::Zero( 7, 7 ); 

    // TODO[TOOL]: Choose the point position on your tool (in TOOL coordinates)
    pointPosition = Eigen::Vector3d( 0.0, 0.0, 0.0 );

    H = Eigen::MatrixXd::Zero( 4, 4 );
    R = Eigen::MatrixXd::Zero( 3, 3 );
    H_ini = Eigen::MatrixXd::Zero( 4, 4 );
    R_ini = Eigen::MatrixXd::Zero( 3, 3 );
    p_ini = Eigen::VectorXd::Zero( 3, 1 );
    p_0_ini = Eigen::VectorXd::Zero( 3, 1 );
    p = Eigen::VectorXd::Zero( 3, 1 );

    J = Eigen::MatrixXd::Zero( 6, 7 );

    Kp = Eigen::MatrixXd::Identity( 3, 3 );                 // Translational stiffness
    Kp = 850 * Kp;  // TODO[GAINS]: Tune translational stiffness [N/m] to your task
    Kr = Eigen::MatrixXd::Identity( 3, 3 );                 // Rotational stiffness
    Kr = 150 * Kr; // TODO[GAINS]: Tune rotational stiffness [N·m/rad] to your task

    // Joint space damping
    Bq = Eigen::MatrixXd::Identity( 7, 7 );
    Bq = 0.5 * Bq; // TODO[GAINS]: Tune joint-space damping

    // ************************************************************
    // AVP streamer
    // ************************************************************

    // Start the streamer thread
    boost::thread(&MyLBRClient::runStreamerThread, this).detach();

    // TODO[PATH]: Point to your Python script location or run externally.
    startPythonScript();

    // Transformation matrices of AVP
    H_avp_rw_ini = Eigen::MatrixXd::Identity( 4, 4 );
    R_avp_rw_ini = Eigen::MatrixXd::Identity( 3, 3 );
    p_avp_rw_ini = Eigen::VectorXd::Zero( 3 );

    p_avp_rw = Eigen::VectorXd::Zero( 3 );

    R_avp_rw = Eigen::MatrixXd::Identity( 3, 3 );

    // # definitions can be found here: https://github.com/Improbable-AI/VisionProTeleop
    matrix_rw = new double[16];                     // wrist

    // ************************************************************
    // INCLUDE FT-SENSOR
    // ************************************************************

    f_ext_ee = Eigen::VectorXd::Zero( 3 );
    m_ext_ee = Eigen::VectorXd::Zero( 3 );
    f_ext = Eigen::VectorXd::Zero( 3 );
    m_ext = Eigen::VectorXd::Zero( 3 );

    // TODO[FT]: Declare/initialize your force-torque sensor and populate ftsSignal in forceSensorThread()

    // Start the force sensor thread
    boost::thread(&MyLBRClient::forceSensorThread, this).detach();

    printf( "Sensor Activated. \n\n" );

    // ************************************************************
    // Initial print
    // ************************************************************

    printf( "Exp[licit](c)-cpp-FRI, https://explicit-robotics.github.io \n\n" );
    printf( "Robot '" );
    printf( "%s", myLBR->Name );
    printf( "' initialised. Ready to rumble! \n\n" );

}


/**
* \brief Destructor
*
*/
MyLBRClient::~MyLBRClient()
{
    // TODO[SHM]: Keep this only if your process owns/creates the segment. Remove if shared with other processes that outlive this object.
    boost::interprocess::shared_memory_object::remove("SharedMemory_AVP");
    delete myLBR; 
}


/**
* \brief Implements an IIR Filter which is used to send the previous joint position to the command function, so that KUKA's internal friction compensation can be activated. The filter function was generated by the application WinFilter (http://www.winfilter.20m.com/).
*
* @param NewSample The current joint position to be provided as input to the filter.
*/
void iir(double NewSample[7])
{
    double ACoef[ NCoef+1 ] = {
        0.05921059165970496400,
        0.05921059165970496400
    };

    double BCoef[ NCoef+1 ] = {
        1.00000000000000000000,
        -0.88161859236318907000
    };

    int n;

    // Shift the old samples
    for ( int i=0; i<7; i++ )
    {
        for( n=NCoef; n>0; n-- )
        {
            filterInput[i][n] = filterInput[i][n-1];
            filterOutput[i][n] = filterOutput[i][n-1];
        }
    }

    // Calculate the new output
    for (int i=0; i<7; i++)
    {
        filterInput[i][0] = NewSample[i];
        filterOutput[i][0] = ACoef[0] * filterInput[i][0];
    }

    for (int i=0; i<7; i++)
    {
        for(n=1; n<=NCoef; n++)
            filterOutput[i][0] += ACoef[n] * filterInput[i][n] - BCoef[n] * filterOutput[i][n];
    }
}

//******************************************************************************
void MyLBRClient::onStateChange(ESessionState oldState, ESessionState newState)
{
    LBRClient::onStateChange(oldState, newState);
    // react on state change events
    switch (newState)
    {
    case MONITORING_WAIT:
    {
        break;
    }
    case MONITORING_READY:
    {
        sampleTime = robotState().getSampleTime();
        break;
    }
    case COMMANDING_WAIT:
    {
        break;
    }
    case COMMANDING_ACTIVE:
    {
        break;
    }
    default:
    {
        break;
    }
    }
}

//******************************************************************************
void MyLBRClient::monitor()
{

    // Copied from FRIClient.cpp
    robotCommand().setJointPosition(robotState().getCommandedJointPosition());

    // Copy measured joint positions (radians) to _qcurr, which is a double
    memcpy( qCurr, robotState().getMeasuredJointPosition(), 7*sizeof(double) );

    // Initialise the q for the previous NCoef timesteps
    for( int i=0; i<NCoef+1; i++ )
    {
        iir(qCurr);
    }
}

//******************************************************************************
void MyLBRClient::waitForCommand()
{
    // If we want to command torques, we have to command them all the time; even in
    // waitForCommand(). This has to be done due to consistency checks. In this state it is
    // only necessary, that some torque values are sent. The LBR does not take the
    // specific value into account.

    if(robotState().getClientCommandMode() == TORQUE){

        robotCommand().setTorque(torques);
        robotCommand().setJointPosition(robotState().getIpoJointPosition());            // Just overlaying same position
    }

}

//******************************************************************************
void MyLBRClient::command()
{

    // ************************************************************
    // Read out relative positions in AVP coordinates

    if(currentTime < sampleTime)
    {
        startPythonScript();
    }

    // Lock mutex and update local variables from shared memory
    double* h_rw;

    dataMutex.lock();
    h_rw = matrix_rw;
    dataMutex.unlock();

    // Convert AVP transformation to Eigen
    Eigen::MatrixXd H_avp_rw = Eigen::Map<Eigen::MatrixXd>(h_rw, 4, 4);

    // Rotation of knuckle with respect to avp
    R_avp_rw = H_avp_rw.transpose().block< 3, 3 >( 0, 0 ); 
    // Position of knuckle with respect to avp
    p_avp_rw = H_avp_rw.transpose().block< 3, 1 >( 0, 3 ); 

    // ****************************************************matrix********
    // Get FTSensor data
    double* fts_bt;

    mutexFTS.lock();
    fts_bt = f_sens_ee; // TODO[FT]: Make sure f_sens_ee points to a valid 6-double buffer {Fx,Fy,Fz,Mx,My,Mz}
    mutexFTS.unlock();

    f_ext_ee[0] = fts_bt[0];
    f_ext_ee[1] = fts_bt[1];
    f_ext_ee[2] = fts_bt[2];
    m_ext_ee[0] = fts_bt[3];
    m_ext_ee[1] = fts_bt[4];
    m_ext_ee[2] = fts_bt[5];

    // Convert to robot base coordinates
    f_ext = R * f_ext_ee;
    m_ext = R * m_ext_ee;

    // ************************************************************
    // Get robot measurements

    memcpy( qOld, qCurr, 7*sizeof(double) );
    memcpy( qCurr, robotState().getMeasuredJointPosition(), 7*sizeof(double) );
    memcpy( tauExternal, robotState().getExternalTorque(), 7*sizeof(double) );

    for (int i=0; i < myLBR->nq; i++)
    {
        q[i] = qCurr[i];
    }

    for (int i=0; i < 7; i++)
    {
        dq[i] = (qCurr[i] - qOld[i]) / sampleTime;
    }

    // ************************************************************
    // Calculate kinematics and dynamics

    // Transformation and Rotation Matrix
    H = myLBR->getForwardKinematics( q, 7, pointPosition );
    R = H.block< 3, 3 >( 0, 0 );
    p = H.block< 3, 1 >( 0, 3 );

    // Transform rotations to quaternions
    Eigen::Quaterniond Q(R);
    Q.normalize();

    // Extract rotation angle
    double theta = 2 * acos( Q.w() );
    double eps = 0.01;
    if( theta <  0.01 ){
        theta = theta + eps ;
    }

    // Compute norm factor, handle edge case for small theta
    double sin_half_theta  = sin(theta  / 2);
    double norm_fact ;
    if (fabs(sin_half_theta ) < 1e-6) {  // Handle small-angle case
        norm_fact  = 1.0;  // Default to 1, or handle separately
    } else {
        norm_fact  = 1.0 / sin_half_theta ;
    }

    Eigen::VectorXd u = Eigen::VectorXd::Zero( 3, 1 );
    u[0] = norm_fact  * Q.x();
    u[1] = norm_fact  * Q.y();
    u[2] = norm_fact  * Q.z();

    //  Get initial transformation of first iteration
    if(currentTime < sampleTime)
    {
        H_ini = H;
        R_ini = R;
        p_ini = p;

        // Get initial AVP transformation
        p_avp_rw_ini = p_avp_rw;
        R_avp_rw_ini = R_avp_rw;

        // Get initial q
        q_ini = q;
    }

    // Jacobian, translational and rotation part
    J = myLBR->getHybridJacobian( q, pointPosition );
    Eigen::MatrixXd J_v = J.block(0, 0, 3, 7);
    Eigen::MatrixXd J_w = J.block(3, 0, 3, 7);

    // Mass matrix
    M = myLBR->getMassMatrix( q );
    Eigen::MatrixXd M_inv = M.inverse();

    // Cartesian mass matrix
    double k = 0.01;                // TODO[REG]: Least-squares regularizer
    Eigen::MatrixXd Lambda = getLambdaLeastSquares(M, J, k);
    Eigen::MatrixXd Lambda_v = getLambdaLeastSquares(M, J_v, k);
    Eigen::MatrixXd Lambda_w = getLambdaLeastSquares(M, J_w, k);

    // ****************** Convert AVP displacement to robot coordinates ******************

    // Displacement from initial position
    Eigen::VectorXd del_p_avp_rw = p_avp_rw - p_avp_rw_ini;

    // Transform to homogeneous coordinates
    Eigen::VectorXd del_p_avp_rw_4d = Eigen::VectorXd::Zero(4, 1);
    del_p_avp_rw_4d[3] = 1;
    del_p_avp_rw_4d.head<3>() = del_p_avp_rw;

    // Transformation to robot base coordinates
    Eigen::MatrixXd H_0_avp = Eigen::MatrixXd::Zero( 4, 4 );
    H_0_avp.block<3, 3>(0, 0) = R_ini;
    H_0_avp.block<3, 1>(0, 3) = p_ini;

    // Extract 3x1 position
    Eigen::VectorXd p_0_4d = H_0_avp * del_p_avp_rw_4d;
    Eigen::VectorXd p_0 = p_0_4d.block<3, 1>(0, 0);


    // ****************** Convert AVP rotation to robot coordinates ******************

    // Change rotation based on Apple Vision Pro
    Eigen::Matrix3d del_R = R_avp_rw_ini.transpose() * R_avp_rw;

    Eigen::Matrix3d R_ee_des =  R.transpose() * R_ini * del_R;

    // Transform rotations to quaternions
    Eigen::Quaterniond Q_ee_des(R_ee_des);
    Q_ee_des.normalize();

    // Extract rotation angle
    double theta_0 = 2 * acos( Q_ee_des.w() );
    double eps_0 = 0.01;
    if( theta_0 <  0.01 ){
        theta_0 = theta_0 + eps_0;
    }

    // Compute norm factor, handle edge case for small theta
    double sin_half_theta_0 = sin(theta_0 / 2);
    double norm_fact_0;
    if (fabs(sin_half_theta_0) < 1e-6) {  // Handle small-angle case
        norm_fact_0 = 1.0;  // Default to 1, or handle separately
    } else {
        norm_fact_0 = 1.0 / sin_half_theta_0;
    }

    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero( 3, 1 );
    u_ee[0] = norm_fact_0 * Q_ee_des.x();
    u_ee[1] = norm_fact_0 * Q_ee_des.y();
    u_ee[2] = norm_fact_0 * Q_ee_des.z();

    // Transform to robot base coordinates
    Eigen::VectorXd u_0 = R * u_ee;

    // ************************************************************
    // Translational task-space impedance controller

    Eigen::VectorXd dx = J_v * dq;
    Eigen::VectorXd del_p = (p_0 - p);

    // Damping design
    double damping_factor_v = 0.7; // TODO[GAINS]: Tune damping factor
    Eigen::Vector3d Kp_diag = Kp.diagonal();
    Eigen::Matrix3d Lambda_v_3d = Lambda_v;
    double lambda_v = compute_lambda(Lambda_v_3d, Kp_diag, damping_factor_v);
    Eigen::MatrixXd Bp = lambda_v * Kp;

    // Calculate force
    Eigen::VectorXd f = Kp * del_p - Bp * dx;

    // Convert to torques
    Eigen::VectorXd tau_translation = J_v.transpose() * f;


    // ************************************************************
    // Rotational task-space impedance controller
    Eigen::VectorXd omega = J_w * dq;

    // Damping design
    double damping_factor_r = 0.7; // TODO[GAINS]: Tune damping factor
    Eigen::Vector3d Kr_diag = Kr.diagonal();
    Eigen::Matrix3d Lambda_w_3d = Lambda_w;
    double lambda_w = compute_lambda(Lambda_w_3d, Kr_diag, damping_factor_r);
    Eigen::MatrixXd Br = lambda_w * Kr;

    // Calculate moment
    Eigen::VectorXd m = Kr * u_0 * theta_0 - Br * omega;

    // Convert to torques
    Eigen::VectorXd tau_rotation = J_w.transpose() * m;


    // ************************************************************
    // Nullspace joint space stiffness
    Eigen::MatrixXd J_bar = M_inv * J.transpose() * Lambda;

    Eigen::MatrixXd N = Eigen::MatrixXd::Identity(7, 7) - J.transpose() * J_bar.transpose();

    Eigen::VectorXd tau_q = -Bq * dq;

    // ************************************************************
    // Control torque
    tau_motion = tau_translation + tau_rotation + (N * tau_q);

    // ************************************************************
    // YOUR CODE ENDS HERE
    // ************************************************************

    // For first iteration
    if( currentTime < sampleTime )
    {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }

    // NOTE[FILTER]: Simple 3-sample smoothing;
    tau_total = ( tau_motion + tau_previous + tau_prev_prev ) / 3;

    for ( int i=0; i<7; i++ )
    {
        qApplied[i] = filterOutput[i][0];
        torques[i] = tau_total[i];
    }

    // Command values (must be double arrays!)
    if (robotState().getClientCommandMode() == TORQUE)
    {
        robotCommand().setJointPosition(qApplied);
        robotCommand().setTorque(torques);
    }

    // IIR filter input
    iir(qCurr);

    // Updates
    if (currentTime == 0.0)
    {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }
    tau_prev_prev = tau_previous;
    tau_previous = tau_motion;
    

    currentTime = currentTime + sampleTime;

}


/**
* \brief Streamer Thread that polls the Vision Pro data and passes it to the command() loop
*
*/
void MyLBRClient::runStreamerThread() {
    try {
        // Create or open shared memory
        // TODO[SHM]: Ensure your Python writer uses the SAME name "SharedMemory_AVP"
        boost::interprocess::shared_memory_object shm(
            boost::interprocess::open_or_create, "SharedMemory_AVP", boost::interprocess::read_write);

        // NOTE[SIZE]: 1*17 doubles + 1 int64_t; verify matches Python layout
        shm.truncate(1 * 17 * sizeof(double) + sizeof(int64_t));

        // Map the shared memory
        boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);

        // Define pointers based on shared memory layout
        int64_t* ready_flag = reinterpret_cast<int64_t*>(region.get_address()); // First 8 bytes
        double* matrix_data_rw = reinterpret_cast<double*>(static_cast<char*>(region.get_address()) + sizeof(int64_t)); // 4x4 matrix [0, :, :]

        // Wait for Python to initialize
        while (*ready_flag == -1) {
            std::cout << "Waiting for Python to initialize...  \n\n" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Python initialized. Starting processing...  \n\n" << std::endl;

        int timeout_counter = 0;
        while (true) { 
            if (*ready_flag == 1) {  // Check if Python has written new data

                dataMutex.lock();

                matrix_rw = matrix_data_rw;
                
                dataMutex.unlock();

                // Reset the flag
                *ready_flag = 0;
                timeout_counter = 0;  // Reset the timeout counter
            } else {
                timeout_counter++;
            }

            if (timeout_counter > 1000) {  // If nothing changes for a while
                timeout_counter = 0;  // Reset timeout to avoid permanent stop
            }

        }

    } catch (const std::exception& e) {
        std::cerr << "Error in shared memory operation: " << e.what() << std::endl;
    }
}



/**
* \brief Opens and runs the python script, stored locally
*
*/
void MyLBRClient::startPythonScript() {
    // TODO[PATH]: Update path to your script or remove if you launch Python externally.
    boost::thread pythonThread([]() {
        const std::string pythonScriptPath = "../VisionProCppCommunication.py";
        const std::string pythonCommand = "python3 " + pythonScriptPath;

        int retCode = system(pythonCommand.c_str());
        if (retCode != 0) {
            std::cerr << "Error: Python script failed with return code " << retCode << std::endl;
        } else {
            std::cout << "Python script executed successfully." << std::endl;
        }
    });
    pythonThread.detach(); 
}


/**
* \brief Thread that polls the force sensor data and passes it to the command() loop
*
*/
void MyLBRClient::forceSensorThread()
{
    while(true){

        // TODO[FT]: Acquire your force sensor data here
        // double* ftsSignal = ... ; // Must point to 6 consecutive doubles {Fx,Fy,Fz,Mx,My,Mz}

        //****************** Update everything at the end with one Mutex ******************//
        mutexFTS.lock();

        f_sens_ee = ftsSignal;

        mutexFTS.unlock();
    }
}


/**
* \brief Function to compute damping factor, applied to stiffness matrix
*
*/
double MyLBRClient::compute_lambda(Eigen::Matrix3d& Lam, Eigen::Vector3d& k_t, double damping_factor)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Lam);
    Eigen::Matrix3d U = solver.eigenvectors();
    Eigen::Matrix3d Sigma = solver.eigenvalues().asDiagonal();

    for(int i=0; i<3; i++)
    {
        Sigma(i,i) = std::sqrt(Sigma(i,i));
    }

    // Compute sqrt(Lambda)
    Eigen::Matrix3d sqrt_Lambda = U * Sigma * U.transpose();

    // Convert k_t to a diagonal matrix
    Eigen::Matrix3d sqrt_k_t = k_t.array().sqrt().matrix().asDiagonal();

    // Compute b_t
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity() * damping_factor;
    Eigen::Matrix3d b_t = sqrt_Lambda * D * sqrt_k_t + sqrt_k_t * D * sqrt_Lambda;

    // Compute lambda
    double lambda = (2.0 * b_t.trace()) / k_t.sum(); 
    return lambda;
}


/**
* \brief Function to compute damping factor, applied to stiffness matrix
*
*/
Eigen::MatrixXd MyLBRClient::getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k)
{

    Eigen::MatrixXd Lam_Inv = J * M.inverse() * J.transpose() + ( k * k ) * Eigen::MatrixXd::Identity( J.rows(), J.rows() );
    Eigen::MatrixXd Lam = Lam_Inv.inverse();

    return Lam;

}