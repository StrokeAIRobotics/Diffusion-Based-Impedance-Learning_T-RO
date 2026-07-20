/**

DISCLAIMER OF WARRANTY

The Software is provided "AS IS" and "WITH ALL FAULTS,"
without warranty of any kind, including without limitation the warranties
of merchantability, fitness for a particular purpose and non-infringement.
KUKA makes no warranty that the Software is free of defects or is suitable
for any particular purpose. In no event shall KUKA be responsible for loss
or damages arising from the installation or use of the Software,
including but not limited to any indirect, punitive, special, incidental
or consequential damages of any character including, without limitation,
damages for loss of goodwill, work stoppage, computer failure or malfunction,
or any and all other commercial damages or losses.
The entire risk to the quality and performance of the Software is not borne by KUKA.
Should the Software prove defective, KUKA is not liable for the entire cost
of any service and repair.


COPYRIGHT

All Rights Reserved
Copyright (C)  2014-2015
KUKA Roboter GmbH
Augsburg, Germany

This material is the exclusive property of KUKA Roboter GmbH and must be returned
to KUKA Roboter GmbH immediately upon request.
This material and the information illustrated or contained herein may not be used,
reproduced, stored in a retrieval system, or transmitted in whole
or in part in any way - electronic, mechanical, photocopying, recording,
or otherwise, without the prior written consent of KUKA Roboter GmbH.


\file
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

    // Time variables for control loop (will be set from FRI)
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

    // Initialize R_ee_fs as a 3x3 matrix
    R_model = Eigen::MatrixXd::Zero( 3, 3 );            // from correct orientation to wrong orientation
    R_model <<  0.0, 1.0,  0.0,
            -1.0, 0.0,  0.0,
            0.0, 0.0,  1.0;

    H_ini = Eigen::MatrixXd::Zero( 4, 4 );
    R_ini = Eigen::MatrixXd::Zero( 3, 3 );
    p_ini = Eigen::VectorXd::Zero( 3, 1 );
    p = Eigen::VectorXd::Zero( 3, 1 );

    J = Eigen::MatrixXd::Zero( 6, 7 );

    Lambda_v = Eigen::MatrixXd::Zero(3, 3);
    Lambda_w = Eigen::MatrixXd::Zero(3, 3);
    dx = Eigen::VectorXd::Zero(3);
    omega = Eigen::VectorXd::Zero(3);

    // TODO[GAINS]: Tune Kp (N/m) and Kr (Nm/rad) for your application
    Kp = Eigen::MatrixXd::Identity(3, 3);
    Kp = 800 * Kp; 
    Kr = Eigen::MatrixXd::Identity(3, 3);
    Kr = 150 * Kr; 

    p_0 = Eigen::Vector3d::Zero(3);
    u_0 = Eigen::Vector3d::Zero(3);

    // TODO[DAMPING]: Tune joint-space damping if needed
    Bq = Eigen::MatrixXd::Identity(7, 7);
    Bq = 0.5 * Bq;


    // ************************************************************
    // LOAD TRAJECTORY
    // ************************************************************
    // TODO[TRAJ]: Put your trajectory filename here (must exist under basePath)
    //std::string selectedTemplate = "parkour_fullTrajectory.txt" ; 
    std::string selectedTemplate = "pegInHole_star_fullTrajectory.txt";     // OR: pegInHole_circular_fullTrajectory.txt, pegInHole_square_fullTrajectory.txt

    // TODO[PATH]: Point this to your data directory (relative or absolute)
    std::string basePath = "../data/";
    std::string filePath;
    filePath = basePath + selectedTemplate;

    std::ifstream infile(filePath);
    if (!infile.is_open()) {
        std::cerr << "Failed to open trajectory file: " << filePath << std::endl;
        std::exit(1);
    } else {
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            double time, x, y, z;
            double r11, r12, r13;
            double r21, r22, r23;
            double r31, r32, r33;

            if (iss >> time >> x >> y >> z
                    >> r11 >> r12 >> r13
                    >> r21 >> r22 >> r23
                    >> r31 >> r32 >> r33) {

                p_0_all.emplace_back(x, y, z);

                Eigen::Matrix3d R_txt;
                R_txt << r11, r12, r13,
                    r21, r22, r23,
                    r31, r32, r33;
                R_0_all.push_back(R_txt);  
            }
        }
        infile.close();
        std::cout << "Trajectory loaded: " << p_0_all.size() << " steps from '" << filePath << "'" << std::endl;
    }

    // ************************************************************
    // SHARED MEMORY
    // ************************************************************
    buffer_ready_for_write.store(false);
    using namespace boost::interprocess;

    // Start the Python script
    // TODO[PY]: Ensure the Python script path in startPythonScript() is correct and accessible
    startPythonScript();


    // ************************************************************
    // Shared Memory for model
    // ************************************************************

    // Wait for 5 seconds to ensure the Python script has time to initialize
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // Retry logic for shared memory initialization (tune if your startup is slower/faster)
    int retry_count = 10; // TODO[IPC]: Increase/decrease retries if needed
    while (retry_count > 0) {
        try {

            // Shared memory for reading (Python ➝ Robot)
            // TODO[IPC]: Must match Python's created name and layout
            shared_memory_object shm_in(open_only, "SharedMemory_1", read_write);
            this->region_in = mapped_region(shm_in, read_write);
            shm_read_buffer = static_cast<const double*>(region_in.get_address());

            // Version flag for synchronization (read shared memory)
            this->version_1_flag = reinterpret_cast<int64_t*>(const_cast<double*>(shm_read_buffer));

            // Shared memory for writing (Robot ➝ Python)
            // TODO[IPC]: Must match Python's created name and layout
            shared_memory_object shm_out(open_only, "SharedMemory_2", read_write);
            this->region_out = mapped_region(shm_out, read_write);

            // Version flag for synchronization (write shared memory)
            void* raw_ptr = region_out.get_address();
            version_2_flag = reinterpret_cast<int64_t*>(raw_ptr);
            shm_write_buffer = reinterpret_cast<double*>((char*)raw_ptr + sizeof(int64_t));

            // Wait for both flags to be initialized properly (Python side sets to 0/1)
            while (*version_1_flag == -1 || *version_2_flag == -1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Check every 100ms
            }

            // Add bounds checking
            if (shm_write_buffer == nullptr || shm_read_buffer == nullptr) {
                std::cerr << "Error: Shared memory buffer is null." << std::endl;
                exit(1);
            }

            break; // Exit retry loop if successful
        } catch (const interprocess_exception& e) {
            std::cerr << "Error initializing shared memory: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait before retrying
            retry_count--;
        }
    }
    
    if (retry_count == 0) {
        std::cerr << "Failed to initialize shared memory after multiple attempts." << std::endl;
        exit(1);
    }

    boost::thread(&MyLBRClient::runSharedMemoryThread, this).detach();
    std::cerr << "Shared memory thread started." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); 

    printf("Shared memory initialized.\n");

    // ************************************************************
    // INCLUDE FT-SENSOR
    // ************************************************************
    // TODO[FT]:  initialize your force-torque sensor here
    f_ext_ee = Eigen::VectorXd::Zero( 3 );
    m_ext_ee = Eigen::VectorXd::Zero( 3 );
    f_ext = Eigen::VectorXd::Zero( 3 );
    f_ext_model = Eigen::VectorXd::Zero( 3 );
    m_ext = Eigen::VectorXd::Zero( 3 );
    m_ext_model = Eigen::VectorXd::Zero( 3 );
    f_ini = Eigen::Vector3d::Zero();
    m_ini = Eigen::Vector3d::Zero();
    
    boost::thread(&MyLBRClient::forceSensorThread, this).detach();

    printf( "Force Sensor Activated. \n\n" );


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
    delete myLBR;
}


/**
* \brief Implements an IIR Filter which is used to send the previous joint position to the command function, so that KUKA's internal friction compensation can be activated. The filter function was generated by the application WinFilter (http://www.winfilter.20m.com/).
*
* @param NewSample The current joint position to be provided as input to the filter.
*/
void MyLBRClient::iir(const double* NewSample)
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
    // only necessary, that some torque vlaues are sent. The LBR does not take the
    // specific value into account.

    if(robotState().getClientCommandMode() == TORQUE){

        robotCommand().setTorque(torques);
        robotCommand().setJointPosition(robotState().getIpoJointPosition());            // Just overlaying same position
    }

}

//******************************************************************************
void MyLBRClient::command()
{

     // ****************************************************
    // Get FTSensor data and process it

    double* fts_bt;

    // Read from force sensor thread
    mutexFTS.lock();
    fts_bt = f_sens_ee;
    mutexFTS.unlock();

    // TODO[FT]: If you stubbed force sensor, ensure fts_bt points to 6 valid doubles {Fx,Fy,Fz,Mx,My,Mz}
    // Assign values
    f_ext_ee[0] = fts_bt[0];
    f_ext_ee[1] = fts_bt[1];
    f_ext_ee[2] = fts_bt[2];
    m_ext_ee[0] = fts_bt[3];
    m_ext_ee[1] = fts_bt[4];
    m_ext_ee[2] = fts_bt[5];

    // Get initial force and moment at force sensor
    if (!init_done) {
        f_ini = f_ext_ee;
        m_ini = m_ext_ee;
        init_done = true;
    }

    // Force and moment for stiffness estimator
    f_ext_ee = f_ext_ee - f_ini; // Subtract initial force through weight of end-effector
    f_ext = R * f_ext_ee; 
    
    m_ext_ee = m_ext_ee - m_ini; // Subtract initial moment through weight of end-effector
    m_ext = R * m_ext_ee;  

    // Corrected force and moment for diffusion model
    Eigen::VectorXd f_ext_ee_model = R_model * f_ext_ee;
    f_ext_model = R * R_model * f_ext_ee;  
    Eigen::VectorXd m_ext_ee_model = R_model * m_ext_ee; 
    m_ext_model = R * R_model * m_ext_ee; 

    // ************************************************************
    // Get robot measurements

    memcpy(qOld, qCurr, 7 * sizeof(double));
    memcpy(qCurr, robotState().getMeasuredJointPosition(), 7 * sizeof(double));
    memcpy(tauExternal, robotState().getExternalTorque(), 7 * sizeof(double));


    for (int i = 0; i < myLBR->nq; i++) {
        q[i] = qCurr[i];
    }

    for (int i = 0; i < 7; i++) {
        dq[i] = (qCurr[i] - qOld[i]) / sampleTime;
    }

    // ************************************************************
    // Calculate kinematics and dynamics
    H = myLBR->getForwardKinematics(q, 7, pointPosition);
    R = H.block<3, 3>(0, 0);
    p = H.block<3, 1>(0, 3);

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
        norm_fact  = 1.0;  
    } else {
        norm_fact  = 1.0 / sin_half_theta ;
    }
   
    Eigen::VectorXd u = Eigen::VectorXd::Zero( 3, 1 );
    u[0] = norm_fact  * Q.x();
    u[1] = norm_fact  * Q.y();
    u[2] = norm_fact  * Q.z(); 

    if (currentTime < sampleTime) {
        H_ini = H;
        R_ini = R;
        p_ini = p;
        q_ini = q;
    }

    J = myLBR->getHybridJacobian(q, pointPosition);
    Eigen::MatrixXd J_v = J.block(0, 0, 3, 7);
    Eigen::MatrixXd J_w = J.block(3, 0, 3, 7);

    M = myLBR->getMassMatrix(q);
    Eigen::MatrixXd M_inv = M.inverse();

    double k = 0.01; // TODO[CTRL]: Tune least-squares reg if needed
    Eigen::MatrixXd Lambda = getLambdaLeastSquares(M, J, k);
    Lambda_v = getLambdaLeastSquares(M, J_v, k);
    Lambda_w = getLambdaLeastSquares(M, J_w, k);
    

    // ************************************************************
    // Read stiffness matrices from shared memory
    shm_read_mutex.lock();
    Kp = Kp_local;
    Kr = Kr_local;
    shm_read_mutex.unlock();

    // ************************************************************
    // Translational task-space impedance controller
    if (step >= p_0_all.size()) {
        step = static_cast<int>(p_0_all.size()) - 1;
    }

    p_0 = p_0_all[step];

    dx = J_v * dq;
    Eigen::VectorXd del_p = (p_0 - p);

    double damping_factor_v = 0.7; // TODO[GAINS]: Tune damping factors if needed
    Eigen::Vector3d Kp_diag = Kp.diagonal();
    double lambda_v = compute_lambda(Lambda_v, Kp_diag, damping_factor_v);
    Eigen::MatrixXd Bp = lambda_v * Kp;

    Eigen::VectorXd f = Kp * del_p - Bp * dx;
    Eigen::VectorXd tau_translation = J_v.transpose() * f;


     // ************************************************************
    // Rotational task-space impedance controller
    
    // Relative rotation from current to desired
    Eigen::Matrix3d R_0_des = R_0_all[step];  // Read from file
    Eigen::Matrix3d R_ee_des = R.transpose() * R_0_des;
    Eigen::Quaterniond Q_0(R_0_des);
    Q_0.normalize();  

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
        norm_fact_0 = 1.0;  
    } else {
        norm_fact_0 = 1.0 / sin_half_theta_0;
    }

    Eigen::VectorXd u_ee = Eigen::VectorXd::Zero( 3, 1 );
    u_ee[0] = norm_fact_0 * Q_ee_des.x();
    u_ee[1] = norm_fact_0 * Q_ee_des.y();
    u_ee[2] = norm_fact_0 * Q_ee_des.z();

    // Transform to robot base coordinates
    Eigen::VectorXd u_0 = R * u_ee;

    // Compute the angular velocity of end-effector
    omega = J_w * dq;

    double damping_factor_r = 0.7; // TODO[GAINS]: Tune damping factors if needed
    Eigen::Vector3d Kr_diag = Kr.diagonal();
    double lambda_w = compute_lambda(Lambda_w, Kr_diag, damping_factor_r);
    Eigen::MatrixXd Br = lambda_w * Kr;

    Eigen::VectorXd m = Kr * u_0 * theta_0 - Br * omega;
    Eigen::VectorXd tau_rotation = J_w.transpose() * m;


    // ************************************************************
    // Nullspace joint space stiffness
    Eigen::MatrixXd J_bar = M_inv * J.transpose() * Lambda;
    Eigen::MatrixXd N = Eigen::MatrixXd::Identity(7, 7) - J.transpose() * J_bar.transpose();
    Eigen::VectorXd tau_q = -Bq * dq;


    // ************************************************************
    // Control torque
    tau_motion = tau_translation + tau_rotation + (N * tau_q);

    if (currentTime < sampleTime) {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }

    tau_total = (tau_motion + tau_previous + tau_prev_prev) / 3;

    for (int i = 0; i < 7; i++) {
        qApplied[i] = filterOutput[i][0];
        torques[i] = tau_total[i];
    }

    if (robotState().getClientCommandMode() == TORQUE) {
        robotCommand().setJointPosition(qApplied);
        robotCommand().setTorque(torques);
    }

    // IIR filter input
    iir(qCurr);

    if (currentTime == 0.0) {
        tau_previous = tau_motion;
        tau_prev_prev = tau_motion;
    }
    tau_previous = tau_motion;
    tau_prev_prev = tau_previous;

    currentTime += sampleTime;
    step++;
}




/**
 * \brief Thread that writes 16 time steps of data to shared memory
 *
 */
void MyLBRClient::runSharedMemoryThread()
{

    std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Wait for Python to initialize

    while (true) { 

        if (version_1_flag == nullptr || version_2_flag == nullptr) {
            std::cerr << "Error: version_x_flag is null!" << std::endl;
            exit(1);
        }

        // Wait for Python to signal readiness
        while (*version_2_flag == -1){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
  
        if (*version_2_flag == 1 && buffer_ready_for_write) {

            int idx = 0;
            buffer_mutex.lock();
            for (int t = 0; t < BUFFER_SIZE; ++t) {
                int i = (buffer_index + t) % BUFFER_SIZE;
            
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = p_buffer[i](j);
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = p0_buffer[i](j);
            
                shm_write_buffer[idx++] = q_buffer[i].x();
                shm_write_buffer[idx++] = q_buffer[i].y();
                shm_write_buffer[idx++] = q_buffer[i].z();
                shm_write_buffer[idx++] = q_buffer[i].w();
                shm_write_buffer[idx++] = q0_buffer[i].x();
                shm_write_buffer[idx++] = q0_buffer[i].y();
                shm_write_buffer[idx++] = q0_buffer[i].z();
                shm_write_buffer[idx++] = q0_buffer[i].w();
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = f_buffer[i](j);
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = f_ext_model[i](j);
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = m_buffer[i](j);
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = m_ext_model[i](j);
            
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        shm_write_buffer[idx++] = Lambda_v_buffer[i](r, c);
            
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = dx_buffer[i](j);
                for (int j = 0; j < 3; ++j) shm_write_buffer[idx++] = omega_buffer[i](j);
            
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        shm_write_buffer[idx++] = Lambda_w_buffer[i](r, c);
            }

            buffer_ready_for_write = false;
            buffer_mutex.unlock();
        }

        *version_2_flag = 0;
        region_out.flush();  // Flush written changes to OS
        
        // // Wait for Python to signal readiness
         while (*version_1_flag == -1){
             std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (*version_1_flag == 1) {
            
             // Use a mutex just for reading
             shm_read_mutex.lock();
             Kp_local = Eigen::Matrix3d::Zero();
             Kr_local = Eigen::Matrix3d::Zero();

             Kp_local(0, 0) = shm_read_buffer[1];
             Kp_local(1, 1) = shm_read_buffer[2];
             Kp_local(2, 2) = shm_read_buffer[3];
             Kr_local(0, 0) = shm_read_buffer[4];
             Kr_local(1, 1) = shm_read_buffer[5];
             Kr_local(2, 2) = shm_read_buffer[6];
             shm_read_mutex.unlock();
             *version_1_flag = 0;
             region_out.flush();  // Flush written changes to OS
        }

    }
}


/**
 * \brief Opens and runs the Python script, stored locally
 *
 */
void MyLBRClient::startPythonScript() {
    boost::thread pythonThread([]() {

        // TODO[PY]: Set the correct path to your Python script
        const std::string pythonScriptPath = "../stiffness_communication.py";
        const std::string pythonCommand = "python3 " + pythonScriptPath;

        // TODO[ENV]: Ensure your Python env/interpreter can import required packages
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