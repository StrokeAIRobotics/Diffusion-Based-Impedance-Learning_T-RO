#ifndef _KUKA_FRI_MY_LBR_CLIENT_H
#define _KUKA_FRI_MY_LBR_CLIENT_H

#include "friLBRClient.h"
#include "exp_robots.h"
#include "AtiForceTorqueSensor.h"

#include <boost/thread.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <pybind11/embed.h>

#include <Eigen/Dense>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <fstream>
#include <vector>

namespace py = pybind11;

class MyLBRClient : public KUKA::FRI::LBRClient
{
public:
    MyLBRClient(double freqHz, double amplitude);
    ~MyLBRClient() override;

    // FRI callbacks
    void onStateChange(KUKA::FRI::ESessionState oldState,
                       KUKA::FRI::ESessionState newState) override;
    void monitor() override;
    void waitForCommand() override;
    void command() override;

private:
    // --- Helpers
    void iir(const double* newSample);
    void forceSensorThread();
    void runSharedMemoryThread();
    void startPythonScript();

    // --- Python (lifetime in this process)
    py::scoped_interpreter guard; // created in .cpp ctor

    // --- Robot model
    iiwa14* myLBR = nullptr;

    // --- Shared memory (Boost.Interprocess)
    boost::interprocess::mapped_region region_in;
    boost::interprocess::mapped_region region_out;

    // Write buffer (Robot → Python)
    double* shm_write_buffer = nullptr;

    // Read buffer (Python → Robot)
    const double* shm_read_buffer = nullptr;

    // Version flags located at the head of the shared regions
    int64_t* version_1_flag = nullptr;
    int64_t* version_2_flag = nullptr;

    // --- I/O buffers for streaming (ring buffer of size 16)
    static const int BUFFER_SIZE = 16;
    int buffer_index = 0;
    int buffer_fill_count = 0;
    std::atomic<bool> buffer_ready_for_write{false};

    Eigen::Vector3d p_buffer[BUFFER_SIZE];
    Eigen::Quaterniond q_buffer[BUFFER_SIZE];
    Eigen::Quaterniond q0_buffer[BUFFER_SIZE];
    Eigen::Vector3d p0_buffer[BUFFER_SIZE];
    Eigen::Vector3d f_buffer[BUFFER_SIZE];
    Eigen::Vector3d f_buffer_wrong[BUFFER_SIZE];
    Eigen::Vector3d m_buffer[BUFFER_SIZE];
    Eigen::Vector3d m_buffer_wrong[BUFFER_SIZE];
    Eigen::Matrix3d Lambda_v_buffer[BUFFER_SIZE];
    Eigen::Matrix3d Lambda_w_buffer[BUFFER_SIZE];
    Eigen::Vector3d dx_buffer[BUFFER_SIZE];
    Eigen::Vector3d omega_buffer[BUFFER_SIZE];
    std::mutex buffer_mutex;

    // --- Robot state/commands
    double torques[7]{};
    double qInitial[7]{};
    double qApplied[7]{};
    double qCurr[7]{};
    double qOld[7]{};
    double tauExternal[7]{};

    // Timing (sample time is obtained from FRI)
    double sampleTime = 0.0;
    double currentTime = 0.0;

    // --- Task selection
    int bodyIndex = 0;                // if you use it in your kinematics
    Eigen::Vector3d pointPosition;    // tool point in tool frame

    // --- State vectors
    Eigen::VectorXd q;
    Eigen::VectorXd q_ini;
    Eigen::VectorXd dq;

    // --- Torques (filters)
    Eigen::VectorXd tau_motion;
    Eigen::VectorXd tau_previous;
    Eigen::VectorXd tau_prev_prev;
    Eigen::VectorXd tau_total;

    // --- Kinematics/Dynamics
    Eigen::MatrixXd M;
    Eigen::MatrixXd M_inv;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    Eigen::MatrixXd R_model;
    Eigen::MatrixXd J;

    // Cartesian mass matrices & velocities
    Eigen::MatrixXd Lambda_v; // 3x3
    Eigen::MatrixXd Lambda_w; // 3x3
    Eigen::VectorXd dx;       // 3x1
    Eigen::VectorXd omega;    // 3x1

    // Initial pose & current pose
    Eigen::MatrixXd H_ini;
    Eigen::MatrixXd R_ini;
    Eigen::VectorXd p_ini;
    Eigen::VectorXd p;

    // Gains
    Eigen::Matrix3d Kp;
    Eigen::Matrix3d Kr;
    Eigen::MatrixXd Bq; 

    // Trajectory
    Eigen::Vector3d p_0, u_0;
    double theta_0 = 0.0;

    std::vector<Eigen::Vector3d>  p_0_all;
    std::vector<Eigen::Matrix3d>  R_0_all;

    // --- Force/Torque sensor
    double* f_sens_ee = nullptr;              // points to {Fx,Fy,Fz,Mx,My,Mz}

    Eigen::VectorXd f_ext_ee;     // 3x1
    Eigen::VectorXd f_ext_model;     // 3x1
    Eigen::VectorXd m_ext_ee;     // 3x1
    Eigen::VectorXd m_ext_model;     // 3x1
    Eigen::VectorXd f_ext;        // 3x1 (base frame)
    Eigen::VectorXd m_ext;        // 3x1 (base frame)

    bool            init_done = false;
    Eigen::Vector3d f_ini = Eigen::Vector3d::Zero();
    Eigen::Vector3d m_ini = Eigen::Vector3d::Zero();

    std::mutex mutexFTS;      // force sensor data lock
    std::mutex shm_mutex;     // shared memory write lock (if needed)
    std::mutex shm_read_mutex;// shared memory read lock

    // Model-written gains (read via shared memory)
    Eigen::Matrix3d Kp_local = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Kr_local = Eigen::Matrix3d::Zero();

    // Damping/gamma design & operational space inertia
    double        compute_lambda(Eigen::Matrix3d& Lambda, Eigen::Vector3d& k_t, double damping_factor);
    Eigen::MatrixXd getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k);

    // Trajectory step
    int step = 0;

};

#endif // _KUKA_FRI_MY_LBR_CLIENT_H