#include "inference_demo/inference_demo.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <sensor_msgs/Joy.h>
#include <sstream>

namespace inference_demo
{

static std::atomic<bool> g_joy_ready(false);
static sensor_msgs::Joy g_joy_msg;
static void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
    g_joy_msg = *msg;
    g_joy_ready.store(true);
}

static unsigned char* loadData(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char* readFileData(const char* filename, int* modelSize)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = loadData(fp, 0, size);
    fclose(fp);

    *modelSize = size;
    return data;
}

InferenceDemo::InferenceDemo(std::shared_ptr<ros::NodeHandle> nh)
    : nh_(nh), quit_(false), stateReceived_(false), imuReceived_(false)
{
    nh_->param<std::string>("model_type", modelType_, "pi_plus");
    nh_->param<std::string>("policy_path", policyPath_, "policy.rknn");
    nh_->param<int>("num_actions", numActions_, 12);
    nh_->param<int>("num_single_obs", numSingleObs_, 36);
    nh_->param<int>("frame_stack", frameStack_, 1);
    nh_->param<double>("rl_ctrl_freq", rlCtrlFreq_, 100.0);
    nh_->param<double>("clip_obs", clipObs_, 18.0);

    nh_->param<double>("cmd_lin_vel_scale", cmdLinVelScale_, 1.0);
    nh_->param<double>("cmd_ang_vel_scale", cmdAngVelScale_, 1.25);
    nh_->param<double>("rbt_lin_pos_scale", rbtLinPosScale_, 1.0);
    nh_->param<double>("rbt_lin_vel_scale", rbtLinVelScale_, 1.0);
    nh_->param<double>("rbt_ang_vel_scale", rbtAngVelScale_, 1.0);
    nh_->param<double>("action_scale", actionScale_, 1.0);

    std::vector<double> clipLower, clipUpper;
    nh_->param<std::vector<double>>("clip_actions_lower", clipLower, std::vector<double>(numActions_, -3.14));
    nh_->param<std::vector<double>>("clip_actions_upper", clipUpper, std::vector<double>(numActions_, 3.14));

    clipActionsLower_.resize(numActions_);
    clipActionsUpper_.resize(numActions_);
    for (int i = 0; i < numActions_; ++i)
    {
        clipActionsLower_[i] = static_cast<float>(clipLower[i]);
        clipActionsUpper_[i] = static_cast<float>(clipUpper[i]);
    }

    robotJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    robotJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    motorJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    motorJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    eulerAngles_ = Eigen::Vector3d::Zero();
    baseAngVel_ = Eigen::Vector3d::Zero();
    command_ = Eigen::Vector3d::Zero();
    action_ = Eigen::VectorXd::Zero(numActions_);

    observations_ = Eigen::VectorXd::Zero(numSingleObs_);
    for (int i = 0; i < frameStack_; ++i)
    {
        histObs_.push_back(Eigen::VectorXd::Zero(numSingleObs_));
    }
    obsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * frameStack_);

    quat_ = Eigen::Quaterniond::Identity();
    
    nh_->param<std::vector<double>>("urdf_dof_pos_offset", urdfOffset_, std::vector<double>(numActions_, 0.0));
    nh_->param<std::vector<int>>("motor_direction", motorDirection_, std::vector<int>(numActions_, 1));
    
    std::vector<int> defaultMap(numActions_);
    for (int i = 0; i < numActions_; ++i) defaultMap[i] = i;
    nh_->param<std::vector<int>>("actual_to_policy_map", actualToPolicyMap_, defaultMap);
    
    if (urdfOffset_.size() != static_cast<size_t>(numActions_)) {
        urdfOffset_.assign(numActions_, 0.0);
    }
    if (motorDirection_.size() != static_cast<size_t>(numActions_)) {
        motorDirection_.assign(numActions_, 1);
    }
    if (actualToPolicyMap_.size() != static_cast<size_t>(numActions_)) {
        actualToPolicyMap_ = defaultMap;
    }
    
    nh_->param<double>("steps_period", stepsPeriod_, 60.0);
    step_ = 0.0;
}

InferenceDemo::~InferenceDemo()
{
    quit_ = true;
    rknn_destroy(ctx_);
}

bool InferenceDemo::init()
{
    std::string topicName = "/" + modelType_ + "_all";
    jointCmdPub_ = nh_->advertise<sensor_msgs::JointState>(topicName, 10);
    
    std::string presetTopic = "/" + modelType_ + "_preset";
    presetPub_ = nh_->advertise<sensor_msgs::JointState>(presetTopic, 10);

    robotStateSub_ = nh_->subscribe("/sim2real_master_node/rbt_state", 10, &InferenceDemo::robotStateCallback, this);
    motorStateSub_ = nh_->subscribe("/sim2real_master_node/mtr_state", 50, &InferenceDemo::motorStateCallback, this);
    imuSub_ = nh_->subscribe("/imu/data", 1, &InferenceDemo::imuCallback, this);
    cmdVelSub_ = nh_->subscribe("/cmd_vel", 10, &InferenceDemo::cmdVelCallback, this);

    std::string joy_topic = "/joy";
    nh_->param<std::string>("joy_topic", joy_topic, joy_topic);
    static ros::Subscriber joy_sub = nh_->subscribe(joy_topic, 10, joyCallback);

    ros::Rate rate(100);
    int timeout = 50;
    while (ros::ok() && (!stateReceived_ || !imuReceived_) && timeout > 0)
    {
        ros::spinOnce();
        rate.sleep();
        timeout--;
    }
    if (!stateReceived_ || !imuReceived_)
    {
        return false;
    }

    if (!loadPolicy())
    {
        return false;
    }
    return true;
}

bool InferenceDemo::loadPolicy()
{
#ifdef PLATFORM_ARM
    int modelSize = 0;
    unsigned char* modelData = readFileData(policyPath_.c_str(), &modelSize);
    if (!modelData)
    {
        return false;
    }
    int ret = rknn_init(&ctx_, modelData, modelSize, 0, nullptr);
    free(modelData);
    if (ret < 0)
    {
        return false;
    }
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_));
    if (ret < 0)
    {
        return false;
    }
    
    memset(rknnInputs_, 0, sizeof(rknnInputs_));
    rknnInputs_[0].index = 0;
    rknnInputs_[0].size = obsInput_.size() * sizeof(float);
    rknnInputs_[0].pass_through = false;
    rknnInputs_[0].type = RKNN_TENSOR_FLOAT32;
    rknnInputs_[0].fmt = RKNN_TENSOR_NHWC;

    memset(rknnOutputs_, 0, sizeof(rknnOutputs_));
    rknnOutputs_[0].want_float = true;
    return true;
#endif
}

void InferenceDemo::updateObservation()
{
    if (observations_.size() != numSingleObs_)
    {
        observations_.resize(numSingleObs_);
    }

    step_ += 1.0 / stepsPeriod_;

    observations_[0] = currentState_ == STANDBY ?  1.0 : std::sin(2 * M_PI * step_);
    observations_[1] = currentState_ == STANDBY ?  -1.0 : std::cos(2 * M_PI * step_);

    double cmd_x = currentState_ == STANDBY ? 0.0 : command_[0];
    double cmd_y = currentState_ == STANDBY ? 0.0 : command_[1];
    double cmd_yaw = currentState_ == STANDBY ? 0.0 : command_[2];
    
    observations_[2] = cmd_x * cmdLinVelScale_ * (cmd_x < 0 ? 0.5 : 1.0);
    observations_[3] = cmd_y * cmdLinVelScale_;
    observations_[4] = cmd_yaw * cmdAngVelScale_;

    observations_.segment(5, numActions_) = robotJointPositions_ * rbtLinPosScale_;

    observations_.segment(17, numActions_) = robotJointVelocities_ * rbtLinVelScale_;

    observations_.segment(29, 3) = baseAngVel_ * rbtAngVelScale_;

    observations_.segment(32, 3) = eulerAngles_;

    for (int i = 0; i < numSingleObs_; ++i)
    {
        observations_[i] = std::clamp(observations_[i], -clipObs_, clipObs_);
    }

    histObs_.push_back(observations_);
    histObs_.pop_front();
}

void InferenceDemo::updateAction()
{
    for (int i = 0; i < frameStack_; ++i)
    {
        obsInput_.block(0, i * numSingleObs_, 1, numSingleObs_) = histObs_[i].transpose();
    }

    std::vector<float> inputData(obsInput_.size());
    for (size_t i = 0; i < obsInput_.size(); ++i) {
        inputData[i] = obsInput_(i);
    }

    rknnInputs_[0].buf = inputData.data();
    rknn_inputs_set(ctx_, ioNum_.n_input, rknnInputs_);
    rknn_run(ctx_, nullptr);
    rknn_outputs_get(ctx_, ioNum_.n_output, rknnOutputs_, nullptr);
    
    float *outputData = static_cast<float *>(rknnOutputs_[0].buf);
    for (int i = 0; i < numActions_; ++i) {
        action_[i] = std::clamp(outputData[i], clipActionsLower_[i], clipActionsUpper_[i]);
    }

    
    rknn_outputs_release(ctx_, ioNum_.n_output, rknnOutputs_);
}

void InferenceDemo::quat2euler()
{
    double x = quat_.x();
    double y = quat_.y();
    double z = quat_.z();
    double w = quat_.w();

    double t0 = 2.0 * (w * x + y * z);
    double t1 = 1.0 - 2.0 * (x * x + y * y);
    double roll = std::atan2(t0, t1);

    double t2 = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
    double pitch = std::asin(t2);

    double t3 = 2.0 * (w * z + x * y);
    double t4 = 1.0 - 2.0 * (y * y + z * z);
    double yaw = std::atan2(t3, t4);

    eulerAngles_ << roll, pitch, yaw;
}

void InferenceDemo::robotStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        return;
    }
    
    for (int i = 0; i < numActions_; ++i)
    {
        robotJointPositions_[i] = msg->position[i];
        if (msg->velocity.size() >= static_cast<size_t>(numActions_))
        {
            robotJointVelocities_[i] = msg->velocity[i];
        }
    }
    
    if (!stateReceived_)
    {
        stateReceived_ = true;
    }
}

void InferenceDemo::motorStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        return;
    }
    
    for (int i = 0; i < numActions_; ++i)
    {
        int policyIdx = actualToPolicyMap_[i];
        if (policyIdx >= 0 && policyIdx < numActions_)
        {
            motorJointPositions_[policyIdx] = msg->position[i] * motorDirection_[i];
            if (msg->velocity.size() >= static_cast<size_t>(numActions_))
            {
                motorJointVelocities_[policyIdx] = msg->velocity[i] * motorDirection_[i];
            }
        }
    }
    
    if (!stateReceived_)
    {
        stateReceived_ = true;
    }
}

void InferenceDemo::imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    quat_.x() = msg->orientation.x;
    quat_.y() = msg->orientation.y;
    quat_.z() = msg->orientation.z;
    quat_.w() = msg->orientation.w;
    quat2euler();
    baseAngVel_[0] = msg->angular_velocity.x; 
    baseAngVel_[1] = msg->angular_velocity.y; 
    baseAngVel_[2] = msg->angular_velocity.z; 
    if (!imuReceived_)
    {
        imuReceived_ = true;
    }
}

void InferenceDemo::cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    command_[0] = msg->linear.x;
    command_[1] = msg->linear.y;
    command_[2] = msg->angular.z;
}

void InferenceDemo::run()
{
    ros::Rate rate(rlCtrlFreq_);
    const char* stateNames[] = {"NOT_READY", "STANDBY", "RUNNING"};
    static ros::Time last_trigger(0);
    
    while (ros::ok() && !quit_)
    {
        ros::spinOnce();
        
        if (currentState_ != NOT_READY) {
            updateObservation();
            updateAction();
        }


        if (g_joy_ready.load()) {
            int axis2 = 2, axis5 = 5, btn_start = 7, btn_lb = 4;
            nh_->param("joy_axis2", axis2, axis2);
            nh_->param("joy_axis5", axis5, axis5);
            nh_->param("joy_button_start", btn_start, btn_start);
            nh_->param("joy_button_lb", btn_lb, btn_lb);

            bool lt_pressed = (axis2 >= 0 && axis2 < (int)g_joy_msg.axes.size()) && (std::abs(g_joy_msg.axes[axis2]) > 0.8);
            bool rt_pressed = (axis5 >= 0 && axis5 < (int)g_joy_msg.axes.size()) && (std::abs(g_joy_msg.axes[axis5]) > 0.8);
            bool start_pressed = (btn_start >= 0 && btn_start < (int)g_joy_msg.buttons.size()) && (g_joy_msg.buttons[btn_start] == 1);
            bool lb_pressed = (btn_lb >= 0 && btn_lb < (int)g_joy_msg.buttons.size()) && (g_joy_msg.buttons[btn_lb] == 1);
            
            bool trigger_reset = lt_pressed && rt_pressed && start_pressed;
            bool trigger_toggle = lt_pressed && rt_pressed && lb_pressed;

            // LT+RT+START: 从 NOT_READY 进入 RL 模式（STANDBY）
            if (trigger_reset && (ros::Time::now() - last_trigger).toSec() > 1.0) {
                if (currentState_ == NOT_READY) {
                    last_trigger = ros::Time::now();
                    
                    double reset_duration = 2.0;
                    nh_->param("reset_duration", reset_duration, reset_duration);
                    
                    sensor_msgs::JointState preset;
                    preset.header.frame_id = "zero";
                    preset.header.stamp.fromSec(reset_duration);
                    
                    presetPub_.publish(preset);
                    ros::Duration(reset_duration).sleep();
                    currentState_ = STANDBY;
                }
            }
            
            if (trigger_toggle && (ros::Time::now() - last_trigger).toSec() > 1.0) {
                if (currentState_ == STANDBY || currentState_ == RUNNING) {
                    last_trigger = ros::Time::now();
                    
                    if (currentState_ == STANDBY) {
                        currentState_ = RUNNING;
                    } else if (currentState_ == RUNNING) {
                        currentState_ = STANDBY;
                    }
                }
            }
        }

        if (currentState_ == NOT_READY) {
            rate.sleep();
            continue;
        }
 
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time(0);
        
        msg.position.resize(22);
        // 根据状态决定 action 缩放因子：RUNNING 时用 actionScale_，其他状态用 0.05
        double scale = (currentState_ == RUNNING) ? actionScale_ : 0.05;
        
        static int debug_count = 0;
        if (++debug_count % 10 == 0) {
            ROS_INFO("[STATE DEBUG] currentState=%s, scale=%.2f (actionScale_=%.2f)", 
                     stateNames[currentState_], scale, actionScale_);
        }
        
        for (int i = 0; i < 12; ++i) {
            msg.position[i] = action_[i] * scale;
        }
        for (int i = 12; i < 22; ++i) {
            msg.position[i] = 0.0;
        }
        
        jointCmdPub_.publish(msg);
        rate.sleep();
    }
}

} // namespace inference_demo

