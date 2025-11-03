#include "inference_demo/inference_demo.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <sensor_msgs/Joy.h>
#include <sstream>

namespace inference_demo
{

// 全局手柄监听（避免修改头文件）
static std::atomic<bool> g_joy_ready(false);
static sensor_msgs::Joy g_joy_msg;
static void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
    g_joy_msg = *msg;
    if (!g_joy_ready.load()) {
        ROS_INFO("First joystick data received! axes=%zu, buttons=%zu", msg->axes.size(), msg->buttons.size());
    }
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
    ROS_INFO("=== InferenceDemo Constructor Started ===");
    nh_->param<std::string>("model_type", modelType_, "pi_plus");
    nh_->param<std::string>("policy_path", policyPath_, "policy.onnx");
    nh_->param<int>("num_actions", numActions_, 12);
    nh_->param<int>("num_single_obs", numSingleObs_, 48);
    nh_->param<int>("frame_stack", frameStack_, 3);
    nh_->param<double>("rl_ctrl_freq", rlCtrlFreq_, 50.0);
    nh_->param<double>("clip_obs", clipObs_, 18.0);
    ROS_INFO("Basic parameters loaded");

    nh_->param<double>("cmd_lin_vel_scale", cmdLinVelScale_, 2.0);
    nh_->param<double>("cmd_ang_vel_scale", cmdAngVelScale_, 0.25);
    nh_->param<double>("rbt_lin_pos_scale", rbtLinPosScale_, 1.0);
    nh_->param<double>("rbt_lin_vel_scale", rbtLinVelScale_, 0.05);
    nh_->param<double>("rbt_ang_vel_scale", rbtAngVelScale_, 0.25);
    ROS_INFO("Scale parameters loaded");

    std::vector<double> clipLower, clipUpper;
    nh_->param<std::vector<double>>("clip_actions_lower", clipLower, std::vector<double>(numActions_, -3.0));
    nh_->param<std::vector<double>>("clip_actions_upper", clipUpper, std::vector<double>(numActions_, 3.0));
    ROS_INFO("Action clipping parameters loaded");

    clipActionsLower_.resize(numActions_);
    clipActionsUpper_.resize(numActions_);
    for (int i = 0; i < numActions_; ++i)
    {
        clipActionsLower_[i] = static_cast<float>(clipLower[i]);
        clipActionsUpper_[i] = static_cast<float>(clipUpper[i]);
    }
    ROS_INFO("Action clipping vectors initialized");

    robotJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    robotJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    motorJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    motorJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    eulerAngles_ = Eigen::Vector3d::Zero();
    baseAngVel_ = Eigen::Vector3d::Zero();
    command_ = Eigen::Vector3d::Zero();
    action_ = Eigen::VectorXd::Zero(numActions_);
    ROS_INFO("State vectors initialized");

    observations_ = Eigen::VectorXd::Zero(numSingleObs_);
    for (int i = 0; i < frameStack_; ++i)
    {
        histObs_.push_back(Eigen::VectorXd::Zero(numSingleObs_));
    }
    obsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * frameStack_);
    ROS_INFO("Observation vectors initialized");

    quat_ = Eigen::Quaterniond::Identity();
    
    nh_->param<std::vector<double>>("urdf_dof_pos_offset", urdfOffset_, std::vector<double>(numActions_, 0.0));
    nh_->param<std::vector<int>>("motor_direction", motorDirection_, std::vector<int>(numActions_, 1));
    
    // 加载关节顺序映射（从actual到policy，用于输入数据重排）
    std::vector<int> defaultMap(numActions_);
    for (int i = 0; i < numActions_; ++i) defaultMap[i] = i;  // 默认恒等映射
    nh_->param<std::vector<int>>("actual_to_policy_map", actualToPolicyMap_, defaultMap);
    
    if (urdfOffset_.size() != static_cast<size_t>(numActions_)) {
        ROS_WARN("urdf_dof_pos_offset size (%zu) != numActions (%d), using defaults", urdfOffset_.size(), numActions_);
        urdfOffset_.assign(numActions_, 0.0);
    }
    if (motorDirection_.size() != static_cast<size_t>(numActions_)) {
        ROS_WARN("motor_direction size (%zu) != numActions (%d), using defaults", motorDirection_.size(), numActions_);
        motorDirection_.assign(numActions_, 1);
    }
    if (actualToPolicyMap_.size() != static_cast<size_t>(numActions_)) {
        ROS_WARN("actual_to_policy_map size (%zu) != numActions (%d), using identity mapping", actualToPolicyMap_.size(), numActions_);
        actualToPolicyMap_ = defaultMap;
    }
    ROS_INFO("URDF offset, motor direction, and joint mapping loaded");
    
    nh_->param<double>("steps_period", stepsPeriod_, 50.0);
    step_ = 0.0;
    isMoving_ = false;
    completingCycle_ = false;
    lastPitch_ = 0.0;
    lastRoll_ = 0.0;
    ROS_INFO("Gait state machine initialized, steps_period=%.1f", stepsPeriod_);
    
    ROS_INFO("=== InferenceDemo Constructor Completed ===");

    ROS_INFO("========================================");
    ROS_INFO("Inference Demo Configuration:");
    ROS_INFO("  Model Type: %s", modelType_.c_str());
    ROS_INFO("  Policy Path: %s", policyPath_.c_str());
    ROS_INFO("  Num Actions: %d", numActions_);
    ROS_INFO("  Num Single Obs: %d", numSingleObs_);
    ROS_INFO("  Frame Stack: %d", frameStack_);
    ROS_INFO("  RL Ctrl Freq: %.1f Hz", rlCtrlFreq_);
    ROS_INFO("========================================");
}

InferenceDemo::~InferenceDemo()
{
    quit_ = true;
    rknn_destroy(ctx_);
    ROS_INFO("InferenceDemo destroyed.");
}

bool InferenceDemo::init()
{
    ROS_INFO("=== InferenceDemo::init() Started ===");
    std::string topicName = "/" + modelType_ + "_all";
    jointCmdPub_ = nh_->advertise<sensor_msgs::JointState>(topicName, 10);
    ROS_INFO("Publisher created for topic: %s", topicName.c_str());
    
    std::string presetTopic = "/" + modelType_ + "_preset";
    presetPub_ = nh_->advertise<sensor_msgs::JointState>(presetTopic, 10);
    ROS_INFO("Publisher created for preset topic: %s", presetTopic.c_str());

    robotStateSub_ = nh_->subscribe("/sim2real_master_node/rbt_state", 10, &InferenceDemo::robotStateCallback, this);
    motorStateSub_ = nh_->subscribe("/sim2real_master_node/mtr_state", 50, &InferenceDemo::motorStateCallback, this);
    imuSub_ = nh_->subscribe("/imu/data", 1, &InferenceDemo::imuCallback, this);
    cmdVelSub_ = nh_->subscribe("/cmd_vel", 10, &InferenceDemo::cmdVelCallback, this);
    ROS_INFO("Subscribers created");

    ROS_INFO("Publishing to: %s", topicName.c_str());
    ROS_INFO("Subscribing to: /sim2real_master_node/rbt_state, /imu/data, /cmd_vel");

    // 监听手柄（可配置话题名，默认 /joy）
    std::string joy_topic = "/joy";
    nh_->param<std::string>("joy_topic", joy_topic, joy_topic);
    static ros::Subscriber joy_sub = nh_->subscribe(joy_topic, 10, joyCallback);
    ROS_INFO("Subscribed to joystick topic: %s", joy_topic.c_str());

    ROS_INFO("Waiting for robot state and IMU data...");
    ros::Rate rate(10);
    int timeout = 50;
    while (ros::ok() && (!stateReceived_ || !imuReceived_) && timeout > 0)
    {
        ros::spinOnce();
        rate.sleep();
        timeout--;
        if (timeout % 10 == 0)
        {
            ROS_INFO("Waiting... timeout=%d, stateReceived=%d, imuReceived=%d", 
                    timeout, stateReceived_, imuReceived_);
        }
    }
    if (!stateReceived_ || !imuReceived_)
    {
        ROS_ERROR("Timeout waiting for robot data! stateReceived=%d, imuReceived=%d", 
                 stateReceived_, imuReceived_);
        return false;
    }
    ROS_INFO("Robot data received successfully");

    if (!loadPolicy())
    {
        ROS_ERROR("Failed to load policy!");
        return false;
    }
    ROS_INFO("=== InferenceDemo::init() Completed Successfully ===");
    return true;
}

bool InferenceDemo::loadPolicy()
{
    ROS_INFO("=== loadPolicy() Started ===");
#ifdef PLATFORM_ARM
    ROS_INFO("Loading RKNN model: %s", policyPath_.c_str());
    int modelSize = 0;
    unsigned char* modelData = readFileData(policyPath_.c_str(), &modelSize);
    if (!modelData)
    {
        ROS_ERROR("Failed to read RKNN model file!");
        return false;
    }
    ROS_INFO("Model file read successfully, size: %d bytes", modelSize);
    int ret = rknn_init(&ctx_, modelData, modelSize, 0, nullptr);
    free(modelData);
    if (ret < 0)
    {
        ROS_ERROR("rknn_init failed, ret=%d", ret);
        return false;
    }
    ROS_INFO("RKNN context initialized successfully");
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_));
    if (ret < 0)
    {
        ROS_ERROR("rknn_query failed, ret=%d", ret);
        return false;
    }
    ROS_INFO("RKNN query successful, inputs:%d, outputs:%d", ioNum_.n_input, ioNum_.n_output);
    
    memset(rknnInputs_, 0, sizeof(rknnInputs_));
    rknnInputs_[0].index = 0;
    rknnInputs_[0].size = obsInput_.size() * sizeof(float);
    rknnInputs_[0].pass_through = false;
    rknnInputs_[0].type = RKNN_TENSOR_FLOAT32;
    rknnInputs_[0].fmt = RKNN_TENSOR_NHWC;
    ROS_INFO("RKNN input tensor configured, size: %zu", obsInput_.size() * sizeof(float));

    memset(rknnOutputs_, 0, sizeof(rknnOutputs_));
    rknnOutputs_[0].want_float = true;
    ROS_INFO("RKNN output tensor configured");
    ROS_INFO("=== loadPolicy() Completed Successfully ===");
    return true;
#endif
}

void InferenceDemo::updateObservation()
{
    ROS_DEBUG("=== updateObservation() Started ===");
    
    // STANDBY 模式下强制静止，不更新步态状态机
    if (currentState_ == STANDBY)
    {
        // 强制重置步态状态机
        isMoving_ = false;
        completingCycle_ = false;
        step_ = 0.0;
    }
    else if (currentState_ == RUNNING)
    {
        // RUNNING 模式下正常运行步态状态机
        const double VEL_THRESHOLD = 0.01;
        bool hasVelocityCommand = (std::abs(command_[0]) > VEL_THRESHOLD ||
                                   std::abs(command_[1]) > VEL_THRESHOLD ||
                                   std::abs(command_[2]) > VEL_THRESHOLD);

        double currentPitch = eulerAngles_[1];
        double currentRoll = eulerAngles_[0];

        bool attitudeExceeded = false;
        if (!isMoving_ && !completingCycle_)
        {
            attitudeExceeded = (std::abs(currentPitch) > 0.18 || std::abs(currentRoll) > 0.18);
        }

        if (!isMoving_ && !completingCycle_)
        {
            if (hasVelocityCommand || attitudeExceeded)
            {
                isMoving_ = true;
                step_ = 0.0;
                lastPitch_ = currentPitch;
                lastRoll_ = currentRoll;
            }
        }
        else if (isMoving_)
        {
            if (!hasVelocityCommand)
            {
                isMoving_ = false;
                completingCycle_ = true;
            }
        }

        if (isMoving_ || completingCycle_)
        {
            step_ += 1.0 / stepsPeriod_;

            if (step_ >= 1.0)
            {
                step_ = 0.0;
                if (completingCycle_)
                {
                    completingCycle_ = false;
                    lastPitch_ = currentPitch;
                    lastRoll_ = currentRoll;
                }
            }
        }
    }

    observations_[0] = eulerAngles_[2];

    observations_.segment(1, 3) = baseAngVel_ * rbtAngVelScale_;

    Eigen::Vector3d gravityVec(0.0, 0.0, -1.0);
    Eigen::Vector4d quatVec(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    observations_.segment(4, 3) = quatRotateInverse(quatVec, gravityVec);

    observations_[7] = command_[0] * cmdLinVelScale_;
    observations_[8] = command_[1] * cmdLinVelScale_;
    observations_[9] = command_[2] * cmdAngVelScale_;

    bool isStationary = (!isMoving_ && !completingCycle_) || (currentState_ == STANDBY);
    observations_[10] = isStationary ? 0.0 : std::sin(2 * M_PI * step_);
    observations_[11] = isStationary ? 1.0 : std::cos(2 * M_PI * step_);

    observations_.segment(12, numActions_) = motorJointPositions_ * rbtLinPosScale_;

    observations_.segment(24, numActions_) = motorJointVelocities_ * rbtLinVelScale_;

    // 观测固定为36维，不包含上一帧动作

    for (int i = 0; i < numSingleObs_; ++i)
    {
        observations_[i] = std::clamp(observations_[i], -clipObs_, clipObs_);
    }

    histObs_.push_back(observations_);
    histObs_.pop_front();
    
    // 打印观测数据（每10次打印一次）
    static int obs_print_count = 0;
    if (++obs_print_count % 10 == 0) {
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(4);
        
        ROS_INFO("[INFERENCE_DEMO OBS] yaw=%.4f", observations_[0]);
        
        line.str(""); line.clear();
        line << "ang_vel (scaled)=[" << observations_[1] << ", " << observations_[2] << ", " << observations_[3] << "]";
        ROS_INFO("%s", line.str().c_str());
        
        line.str(""); line.clear();
        line << "ang_vel (raw)=[" << baseAngVel_[0] << ", " << baseAngVel_[1] << ", " << baseAngVel_[2] << "]";
        ROS_INFO("%s", line.str().c_str());
        
        line.str(""); line.clear();
        line << "gravity=[" << observations_[4] << ", " << observations_[5] << ", " << observations_[6] << "]";
        ROS_INFO("%s", line.str().c_str());
        
        line.str(""); line.clear();
        line << "cmd=[" << observations_[7] << ", " << observations_[8] << ", " << observations_[9] << "]";
        ROS_INFO("%s", line.str().c_str());
        
        ROS_INFO("phase=[%.4f, %.4f] (sin, cos)", observations_[10], observations_[11]);
        
        line.str(""); line.clear();
        line << "joint_pos=[";
        for (int i = 0; i < 12; ++i) {
            if (i) line << ", ";
            line << observations_[12 + i];
        }
        line << "]";
        ROS_INFO("%s", line.str().c_str());
        
        line.str(""); line.clear();
        line << "joint_vel=[";
        for (int i = 0; i < 12; ++i) {
            if (i) line << ", ";
            line << observations_[24 + i];
        }
        line << "]";
        ROS_INFO("%s", line.str().c_str());
    }
    
    ROS_DEBUG("=== updateObservation() Completed ===");
}

void InferenceDemo::updateAction()
{
    ROS_DEBUG("=== updateAction() Started ===");
    
    // 准备输入数据
    for (int i = 0; i < frameStack_; ++i)
    {
        obsInput_.block(0, i * numSingleObs_, 1, numSingleObs_) = histObs_[i].transpose();
    }
    ROS_DEBUG("Input tensor prepared");

    // 获取输入张量
    std::vector<float> inputData(obsInput_.size());
    for (size_t i = 0; i < obsInput_.size(); ++i) {
        inputData[i] = obsInput_(i); // 转换 obsInput_ 到向量
    }
    ROS_DEBUG("Input data converted to float vector");

    rknnInputs_[0].buf = inputData.data();

    int ret = rknn_inputs_set(ctx_, ioNum_.n_input, rknnInputs_);
    if (ret != RKNN_SUCC) {
        ROS_ERROR_THROTTLE(1.0, "Failed to set RKNN input! ret=%d", ret);
        return;
    }
    ROS_DEBUG("RKNN inputs set successfully");

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        ROS_ERROR_THROTTLE(1.0, "Failed to run RKNN inference! ret=%d", ret);
        return;
    }
    ROS_DEBUG("RKNN inference completed");

    ret = rknn_outputs_get(ctx_, ioNum_.n_output, rknnOutputs_, nullptr);
    if (ret != RKNN_SUCC) {
        ROS_ERROR_THROTTLE(1.0, "Failed to get RKNN output! ret=%d", ret);
        return;
    }
    ROS_DEBUG("RKNN outputs retrieved");
    
    float *outputData = static_cast<float *>(rknnOutputs_[0].buf);
    for (int i = 0; i < numActions_; ++i) {
        float lower_bound = static_cast<float>(clipActionsLower_[i]);
        float upper_bound = static_cast<float>(clipActionsUpper_[i]);
        float clamped_value = std::clamp(outputData[i], lower_bound, upper_bound);
        action_[i] = clamped_value;
    }
    ROS_DEBUG("Actions processed and clamped");
    
    // 打印策略输出结果
    static int print_count = 0;
    if (++print_count % 10 == 0) {  // 每10次打印一次，避免刷屏
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(4);
        line << "[POLICY OUTPUT] action=[";
        for (int i = 0; i < numActions_; ++i) {
            if (i) line << ", ";
            line << action_[i];
        }
        line << "]";
        ROS_INFO("%s", line.str().c_str());
    }
    
    rknn_outputs_release(ctx_, ioNum_.n_output, rknnOutputs_);
    ROS_DEBUG("=== updateAction() Completed ===");
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

Eigen::Vector3d InferenceDemo::quatRotateInverse(const Eigen::Vector4d& q, const Eigen::Vector3d& v)
{
    double qw = q[3];
    Eigen::Vector3d qVec = q.head<3>();

    Eigen::Vector3d a = v * (2.0 * qw * qw - 1.0);
    Eigen::Vector3d b = qVec.cross(v) * qw * 2.0;
    Eigen::Vector3d c = qVec * (qVec.dot(v) * 2.0);

    return a - b + c;
}

void InferenceDemo::robotStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    ROS_DEBUG("Robot state callback triggered");
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        ROS_WARN_THROTTLE(1.0, "Robot joint state size too small: %zu < %d", msg->position.size(), numActions_);
        return;
    }
    
    // 保存robot关节数据（相对位置，来自rbt_state话题）
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
        ROS_INFO("First robot state received. Using first %d joints out of %zu total joints.", 
                numActions_, msg->position.size());
    }
    ROS_DEBUG("Robot state processed successfully");
}

void InferenceDemo::motorStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    ROS_DEBUG("Motor state callback triggered");
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        ROS_WARN_THROTTLE(1.0, "Motor state size too small: %zu < %d", msg->position.size(), numActions_);
        return;
    }
    
    // 保存motor关节数据（绝对角度，来自mtr_state话题）
    // 1. 应用 motor_direction 来转换极性
    // 2. 应用 actual_to_policy_map 来重排输入顺序，确保与策略训练时的关节顺序一致
    //    (例如: mtr_state是左右顺序，policy训练时是右左顺序)
    for (int i = 0; i < numActions_; ++i)
    {
        // actualToPolicyMap_[i] 表示: actual的第i个关节 应该放到 policy的哪个位置
        int policyIdx = actualToPolicyMap_[i];
        if (policyIdx >= 0 && policyIdx < numActions_)
        {
            // 应用电机方向（符号）并存储到policy顺序的位置
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
        ROS_INFO("First motor state received. Applying motor_direction and input joint order mapping.");
    }
    ROS_DEBUG("Motor state processed successfully");
}

void InferenceDemo::imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    ROS_DEBUG("IMU callback triggered");
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
        ROS_INFO("First IMU data received.");
    }
    ROS_DEBUG("IMU data processed successfully");
}

void InferenceDemo::cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    ROS_DEBUG("Command velocity callback triggered");
    command_[0] = msg->linear.x;
    command_[1] = msg->linear.y;
    command_[2] = msg->angular.z;
    ROS_DEBUG("Command velocity: [%.3f, %.3f, %.3f]", command_[0], command_[1], command_[2]);
}

void InferenceDemo::run()
{
    ros::Rate rate(rlCtrlFreq_);
    ROS_INFO("=== Starting inference loop at %.1f Hz ===", rlCtrlFreq_);
    
    const char* stateNames[] = {"NOT_READY", "STANDBY", "RUNNING"};
    ROS_INFO("Initial state: %s (press LT+RT+START to reset and enter STANDBY)", stateNames[currentState_]);
    
    int loopCount = 0;
    static ros::Time last_trigger(0);
    
    while (ros::ok() && !quit_)
    {
        ROS_DEBUG("=== Loop iteration %d, state=%s ===", loopCount++, stateNames[currentState_]);
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
            
            static int debug_count = 0;
            if (++debug_count % 50 == 0) {
                ROS_INFO("State=%s | LT=%.2f RT=%.2f START=%d LB=%d | Reset=%d Toggle=%d",
                    stateNames[currentState_],
                    (axis2 >= 0 && axis2 < (int)g_joy_msg.axes.size()) ? g_joy_msg.axes[axis2] : -999.0,
                    (axis5 >= 0 && axis5 < (int)g_joy_msg.axes.size()) ? g_joy_msg.axes[axis5] : -999.0,
                    start_pressed, lb_pressed, trigger_reset, trigger_toggle);
            }

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
                    ROS_INFO("=== LT+RT+START: Sending ZERO command (%.1fs), entering RL mode ===", reset_duration);
                    
                    ros::Duration(reset_duration).sleep();
                    currentState_ = STANDBY;
                    ROS_INFO("=== Entered RL mode: %s (press LT+RT+LB to toggle RUNNING) ===", stateNames[currentState_]);
                } else {
                    ROS_WARN("LT+RT+START only works in NOT_READY state. Current state: %s", stateNames[currentState_]);
                }
            }
            
            // LT+RT+LB: 在 STANDBY 和 RUNNING 之间切换（仅在 RL 模式下）
            if (trigger_toggle && (ros::Time::now() - last_trigger).toSec() > 1.0) {
                if (currentState_ == STANDBY || currentState_ == RUNNING) {
                    last_trigger = ros::Time::now();
                    
                    if (currentState_ == STANDBY) {
                        currentState_ = RUNNING;
                        ROS_INFO("=== LT+RT+LB: %s -> %s (policy inference enabled) ===", 
                                stateNames[STANDBY], stateNames[RUNNING]);
                    } else if (currentState_ == RUNNING) {
                        currentState_ = STANDBY;
                        ROS_INFO("=== LT+RT+LB: %s -> %s (policy inference disabled, legs send offset only) ===", 
                                stateNames[RUNNING], stateNames[STANDBY]);
                    }
                } else {
                    ROS_WARN("LT+RT+LB only works in RL mode (STANDBY/RUNNING). Current state: %s", stateNames[currentState_]);
                }
            }
        } else {
            static int no_joy_count = 0;
            if (++no_joy_count % 100 == 0) {
                ROS_WARN("Joystick not ready! State=%s", stateNames[currentState_]);
            }
        }

        if (currentState_ == NOT_READY) {
            rate.sleep();
            continue;
        }
 
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time(0);
        
       

        if (currentState_ == STANDBY) {
            msg.position.resize(22);
            for (int i = 0; i < 12; ++i) {
                msg.position[i] = action_[i];
            }
            for (int i = 12; i < 22; ++i) {
                msg.position[i] = 0.0;
            }
        } else if (currentState_ == RUNNING) {
            msg.position.resize(22);
            for (int i = 0; i < 12; ++i) {
                msg.position[i] = action_[i];
            }
            for (int i = 12; i < 22; ++i) {
                msg.position[i] = 0.0;
            }
        }
        
        jointCmdPub_.publish(msg);
        rate.sleep();
    }
    ROS_INFO("=== Inference loop stopped ===");
}

} // namespace inference_demo

