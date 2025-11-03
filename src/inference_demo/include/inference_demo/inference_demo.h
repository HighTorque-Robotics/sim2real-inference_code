#ifndef INFERENCE_DEMO_H
#define INFERENCE_DEMO_H

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <Eigen/Dense>
#include <memory>
#include <deque>
#include <vector>

#ifdef PLATFORM_ARM
#include "rknn_api.h"
#endif

namespace inference_demo
{

class InferenceDemo
{
public:
    InferenceDemo(std::shared_ptr<ros::NodeHandle> nh);
    ~InferenceDemo();

    bool init();
    void run();
    void stop() { quit_ = true; }

private:
    bool loadPolicy();
    void updateObservation();
    void updateAction();
    void quat2euler();
    Eigen::Vector3d quatRotateInverse(const Eigen::Vector4d& q, const Eigen::Vector3d& v);

    void robotStateCallback(const sensor_msgs::JointState::ConstPtr& msg);
    void motorStateCallback(const sensor_msgs::JointState::ConstPtr& msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg);

    std::shared_ptr<ros::NodeHandle> nh_;

    ros::Publisher jointCmdPub_;
    ros::Publisher presetPub_;
    ros::Subscriber robotStateSub_;
    ros::Subscriber motorStateSub_;
    ros::Subscriber imuSub_;
    ros::Subscriber cmdVelSub_;

    std::string modelType_;
    std::string policyPath_;
    int numActions_;
    int numSingleObs_;
    int frameStack_;
    double rlCtrlFreq_;
    double clipObs_;

    double cmdLinVelScale_;
    double cmdAngVelScale_;
    double rbtLinPosScale_;
    double rbtLinVelScale_;
    double rbtAngVelScale_;

    std::vector<float> clipActionsLower_;
    std::vector<float> clipActionsUpper_;

    // Robot关节数据（相对位置，来自rbt_state话题）
    Eigen::VectorXd robotJointPositions_;
    Eigen::VectorXd robotJointVelocities_;
    
    // Motor关节数据（绝对角度，来自mtr_state话题，用于策略推理）
    Eigen::VectorXd motorJointPositions_;
    Eigen::VectorXd motorJointVelocities_;
    
    Eigen::Quaterniond quat_;
    Eigen::Vector3d eulerAngles_;
    Eigen::Vector3d baseAngVel_;

    Eigen::Vector3d command_;

    Eigen::VectorXd observations_;
    std::deque<Eigen::VectorXd> histObs_;
    Eigen::MatrixXd obsInput_;

    Eigen::VectorXd action_;

    std::vector<double> urdfOffset_;
    std::vector<int> motorDirection_;
    std::vector<int> actualToPolicyMap_;  // 输入关节顺序映射：从mtr_state顺序到policy训练时的顺序

    double stepsPeriod_;
    double step_;
    bool isMoving_;
    bool completingCycle_;
    double lastPitch_;
    double lastRoll_;

    bool quit_;
    bool stateReceived_;
    bool imuReceived_;

#ifdef PLATFORM_ARM
    rknn_context ctx_{};
    rknn_input_output_num ioNum_{};
    rknn_input rknnInputs_[1]{};
    rknn_output rknnOutputs_[1]{};
#endif

    // 状态机（类内成员）
    enum State { NOT_READY, STANDBY, RUNNING };
    State currentState_ = NOT_READY;
};

} // namespace inference_demo

#endif

