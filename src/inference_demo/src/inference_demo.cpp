#include "inference_demo/inference_demo.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace inference_demo
{

#ifdef PLATFORM_ARM
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
#endif

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

    jointPositions_ = Eigen::VectorXd::Zero(numActions_);
    jointVelocities_ = Eigen::VectorXd::Zero(numActions_);
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
#ifdef PLATFORM_ARM
    rknn_destroy(ctx_);
#endif
    ROS_INFO("InferenceDemo destroyed.");
}

bool InferenceDemo::init()
{
    ROS_INFO("=== InferenceDemo::init() Started ===");
    std::string topicName = "/" + modelType_ + "_all";
    jointCmdPub_ = nh_->advertise<sensor_msgs::JointState>(topicName, 10);
    ROS_INFO("Publisher created for topic: %s", topicName.c_str());

    robotStateSub_ = nh_->subscribe("/sim2real_master_node/rbt_state", 10, &InferenceDemo::robotStateCallback, this);
    imuSub_ = nh_->subscribe("/imu/data", 1, &InferenceDemo::imuCallback, this);
    cmdVelSub_ = nh_->subscribe("/cmd_vel", 10, &InferenceDemo::cmdVelCallback, this);
    ROS_INFO("Subscribers created");

    ROS_INFO("Publishing to: %s", topicName.c_str());
    ROS_INFO("Subscribing to: /sim2real_master_node/rbt_state, /imu/data, /cmd_vel");

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
    ROS_INFO("RKNN input tensor configured, size: %d", obsInput_.size() * sizeof(float));

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
    
    // 0-2 base_ang_vel
    observations_.segment(0, 3) = baseAngVel_ * rbtAngVelScale_;
    ROS_DEBUG("Base angular velocity processed");

    // 3-5 projected_gravity
    Eigen::Vector3d gravity(0, 0, -1);
    Eigen::Vector4d q(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    observations_.segment(3, 3) = quatRotateInverse(q, gravity);
    ROS_DEBUG("Gravity vector processed");

    // 6,7,8 commands
    observations_[6] = command_[0] * cmdLinVelScale_;
    observations_[7] = command_[1] * cmdLinVelScale_;
    observations_[8] = command_[2] * cmdAngVelScale_;
    ROS_DEBUG("Command velocities processed");

    // 9 standing_command_mask
    if (std::abs(observations_[6]) > 1e-9 || std::abs(observations_[7]) > 1e-9 || std::abs(observations_[8]) > 1e-9)
    {
        observations_[9] = 0.0;
    }
    else
    {
        observations_[9] = 0.0;
    }

    // 10,11 time features (sin/cos)
    observations_[10] = 0.0;
    observations_[11] = 0.0;
    ROS_DEBUG("Command flags and time features processed");

    // 12-23 joint positions
    observations_.segment(12, numActions_) = jointPositions_ * rbtLinPosScale_;
    // 24-35 joint velocities  
    observations_.segment(24, numActions_) = jointVelocities_ * rbtLinVelScale_;
    // 36-47 previous actions
    observations_.segment(36, numActions_) = action_;
    ROS_DEBUG("Joint states and actions processed");

    // Clip observations
    for (int i = 0; i < numSingleObs_; ++i)
    {
        observations_[i] = std::clamp(observations_[i], -clipObs_, clipObs_);
    }
    ROS_DEBUG("Observations clipped");

    histObs_.push_back(observations_);
    histObs_.pop_front();
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
        ROS_WARN_THROTTLE(1.0, "Joint state size too small: %zu < %d", msg->position.size(), numActions_);
        return;
    }
    
    // 只使用前 numActions_ 个关节（腿部关节）
    for (int i = 0; i < numActions_; ++i)
    {
        jointPositions_[i] = msg->position[i];
        if (msg->velocity.size() >= static_cast<size_t>(numActions_))
        {
            jointVelocities_[i] = msg->velocity[i];
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
    int loopCount = 0;
    while (ros::ok() && !quit_)
    {
        ROS_DEBUG("=== Loop iteration %d ===", loopCount++);
        ros::spinOnce();
        updateObservation();
        updateAction();
        
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time(0);
        msg.position.resize(numActions_);
        for (int i = 0; i < numActions_; ++i)
        {
            msg.position[i] = action_[i];
        }
        jointCmdPub_.publish(msg);
        ROS_DEBUG("Joint command published");
        
        rate.sleep();
    }
    ROS_INFO("=== Inference loop stopped ===");
}

} // namespace inference_demo

