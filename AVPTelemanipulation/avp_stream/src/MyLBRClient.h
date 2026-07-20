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



/**
\version {1.9}
*/
#ifndef _KUKA_FRI_MY_LBR_CLIENT_H
#define _KUKA_FRI_MY_LBR_CLIENT_H

#include "friLBRClient.h"
#include "exp_robots.h"

#include <boost/thread.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

using namespace KUKA::FRI;
namespace py = pybind11;

/**
 * \brief Template client implementation.
 */
class MyLBRClient : public LBRClient
{
public:
    MyLBRClient(double freqHz, double amplitude);
    ~MyLBRClient();

    // FRI callbacks
    void onStateChange(ESessionState oldState, ESessionState newState) override;
    void monitor() override;
    void waitForCommand() override;
    void command() override;

private:
    // Python Integration
    py::scoped_interpreter guard;

    // Shared Memory and Threading
    boost::thread streamerThread;
    boost::mutex dataMutex;
    void runStreamerThread();
    void startPythonScript();

    // AVP shared-memory payload (4x4 transform matrix, mapped externally)
    double* matrix_rw;

    // AVP initial and current pose (robot/world aligned versions)
    Eigen::MatrixXd H_avp_rw_ini;
    Eigen::MatrixXd R_avp_rw_ini;
    Eigen::VectorXd p_avp_rw_ini;

    Eigen::MatrixXd R_avp_rw;
    Eigen::VectorXd p_avp_rw;

    // Robot model
    iiwa14* myLBR;

    // Joint arrays used by FRI (raw doubles)
    double torques[7];
    double qInitial[7];
    double qApplied[7];
    double qCurr[7];
    double qOld[7];
    double tauExternal[7];

    // Timing
    double sampleTime;
    double currentTime;

    // Control point on tool
    Eigen::Vector3d pointPosition;

    // State (Eigen vectors)
    Eigen::VectorXd q;
    Eigen::VectorXd q_ini;
    Eigen::VectorXd dq;

    // Torques (Eigen)
    Eigen::VectorXd tau_motion;
    Eigen::VectorXd tau_previous;
    Eigen::VectorXd tau_prev_prev;
    Eigen::VectorXd tau_total;

    // Controller-related matrices
    Eigen::MatrixXd M;
    Eigen::MatrixXd M_inv;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    Eigen::MatrixXd J;

    Eigen::MatrixXd H_ini;
    Eigen::MatrixXd R_ini;
    Eigen::VectorXd p_ini;
    Eigen::VectorXd p;
    Eigen::VectorXd p_0_ini;

    Eigen::MatrixXd Kp;
    Eigen::MatrixXd Kr;
    Eigen::MatrixXd Bq;

    // Force-Torque data containers (populated by your sensor thread)
    boost::thread ftsThread;
    boost::mutex mutexFTS;
    void forceSensorThread();

    double* f_sens_ee;          // expects 6 doubles: {Fx,Fy,Fz,Mx,My,Mz}
    Eigen::VectorXd f_ext_ee;   // forces at EE in EE frame
    Eigen::VectorXd m_ext_ee;   // moments at EE in EE frame
    Eigen::VectorXd f_ext;      // transformed to base
    Eigen::VectorXd m_ext;      // transformed to base

    // Damping design utilities
    double compute_lambda(Eigen::Matrix3d& Lambda, Eigen::Vector3d& k_t, double damping_factor);
    Eigen::MatrixXd getLambdaLeastSquares(Eigen::MatrixXd M, Eigen::MatrixXd J, double k);
};

#endif // _KUKA_FRI_MY_LBR_CLIENT_H