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
    nh_->param<std::string>("model_type", modelType_, "pi_plus");
    nh_->param<std::string>("policy_path", policyPath_, "policy.onnx");
    nh_->param<int>("num_actions", numActions_, 12);
    nh_->param<int>("num_single_obs", numSingleObs_, 48);
    nh_->param<int>("frame_stack", frameStack_, 3);
    nh_->param<double>("rl_ctrl_freq", rlCtrlFreq_, 50.0);
    nh_->param<double>("clip_obs", clipObs_, 18.0);

    nh_->param<double>("cmd_lin_vel_scale", cmdLinVelScale_, 2.0);
    nh_->param<double>("cmd_ang_vel_scale", cmdAngVelScale_, 0.25);
    nh_->param<double>("rbt_lin_pos_scale", rbtLinPosScale_, 1.0);
    nh_->param<double>("rbt_lin_vel_scale", rbtLinVelScale_, 0.05);
    nh_->param<double>("rbt_ang_vel_scale", rbtAngVelScale_, 0.25);

    std::vector<double> clipLower, clipUpper;
    nh_->param<std::vector<double>>("clip_actions_lower", clipLower, std::vector<double>(numActions_, -3.0));
    nh_->param<std::vector<double>>("clip_actions_upper", clipUpper, std::vector<double>(numActions_, 3.0));

    clipActionsLower_.resize(numActions_);
    clipActionsUpper_.resize(numActions_);
    for (int i = 0; i < numActions_; ++i)
    {
        clipActionsLower_[i] = static_cast<float>(clipLower[i]);
        clipActionsUpper_[i] = static_cast<float>(clipUpper[i]);
    }

    jointPositions_ = Eigen::VectorXd::Zero(numActions_);
    jointVelocities_ = Eigen::VectorXd::Zero(numActions_);
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
    std::string topicName = "/" + modelType_ + "_all";
    jointCmdPub_ = nh_->advertise<sensor_msgs::JointState>(topicName, 10);

    robotStateSub_ = nh_->subscribe("rbt_state", 10, &InferenceDemo::robotStateCallback, this);
    imuSub_ = nh_->subscribe("/imu/data", 1, &InferenceDemo::imuCallback, this);
    cmdVelSub_ = nh_->subscribe("/cmd_vel", 10, &InferenceDemo::cmdVelCallback, this);

    ROS_INFO("Publishing to: %s", topicName.c_str());
    ROS_INFO("Subscribing to: rbt_state, /imu/data, /cmd_vel");

    ROS_INFO("Waiting for robot state and IMU data...");
    ros::Rate rate(10);
    int timeout = 50;
    while (ros::ok() && (!stateReceived_ || !imuReceived_) && timeout > 0)
    {
        ros::spinOnce();
        rate.sleep();
        timeout--;
    }
    if (!stateReceived_ || !imuReceived_)
    {
        ROS_ERROR("Timeout waiting for robot data!");
        return false;
    }

    if (!loadPolicy())
    {
        ROS_ERROR("Failed to load policy!");
        return false;
    }
    return true;
}

bool InferenceDemo::loadPolicy()
{
#ifdef PLATFORM_X86_64
    try
    {
        ROS_INFO("Loading OpenVINO model: %s", policyPath_.c_str());
        model_ = core_.read_model(policyPath_);
        compiledModel_ = core_.compile_model(model_, "CPU");
        inferRequest_ = compiledModel_.create_infer_request();
        ROS_INFO("OpenVINO model loaded successfully.");
        return true;
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Failed to load OpenVINO model: %s", e.what());
        return false;
    }
#elif defined(PLATFORM_ARM)
    ROS_INFO("Loading RKNN model: %s", policyPath_.c_str());
    int modelSize = 0;
    unsigned char* modelData = readFileData(policyPath_.c_str(), &modelSize);
    if (!modelData)
    {
        ROS_ERROR("Failed to read RKNN model file!");
        return false;
    }
    int ret = rknn_init(&ctx_, modelData, modelSize, 0, nullptr);
    free(modelData);
    if (ret < 0)
    {
        ROS_ERROR("rknn_init failed, ret=%d", ret);
        return false;
    }
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_));
    if (ret < 0)
    {
        ROS_ERROR("rknn_query failed, ret=%d", ret);
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
    ROS_INFO("RKNN model loaded successfully. in:%d out:%d", ioNum_.n_input, ioNum_.n_output);
    return true;
#else
    ROS_ERROR("No inference engine defined! Please compile with PLATFORM_X86_64 or PLATFORM_ARM");
    return false;
#endif
}

void InferenceDemo::updateObservation()
{
    observations_.segment(0, 3) = baseAngVel_ * rbtAngVelScale_;

    Eigen::Vector3d gravity(0, 0, -1);
    Eigen::Vector4d q(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    observations_.segment(3, 3) = quatRotateInverse(q, gravity);

    observations_[6] = command_[0] * cmdLinVelScale_;
    observations_[7] = command_[1] * cmdLinVelScale_;
    observations_[8] = command_[2] * cmdAngVelScale_;

    observations_[9] = (std::abs(observations_[6]) > 1e-9 || std::abs(observations_[7]) > 1e-9 || std::abs(observations_[8]) > 1e-9) ? 0.0 : 0.0;

    observations_[10] = 0.0;
    observations_[11] = 0.0;

    observations_.segment(12, numActions_) = jointPositions_ * rbtLinPosScale_;
    observations_.segment(24, numActions_) = jointVelocities_ * rbtLinVelScale_;
    observations_.segment(36, numActions_) = action_;

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

#ifdef PLATFORM_X86_64
    auto inputPort = compiledModel_.input();
    ov::Tensor inputTensor = inferRequest_.get_tensor(inputPort);
    float* inputData = inputTensor.data<float>();
    for (size_t i = 0; i < inputTensor.get_size(); ++i)
    {
        inputData[i] = static_cast<float>(obsInput_(i));
    }
    inferRequest_.infer();
    auto outputPort = compiledModel_.output();
    ov::Tensor outputTensor = inferRequest_.get_tensor(outputPort);
    const float* outputData = outputTensor.data<const float>();
    for (int i = 0; i < numActions_; ++i)
    {
        float clampedValue = std::clamp(outputData[i], clipActionsLower_[i], clipActionsUpper_[i]);
        action_[i] = clampedValue;
    }
#elif defined(PLATFORM_ARM)
    std::vector<float> inputData(obsInput_.size());
    for (size_t i = 0; i < obsInput_.size(); ++i)
    {
        inputData[i] = static_cast<float>(obsInput_(i));
    }
    rknnInputs_[0].buf = inputData.data();
    int ret = rknn_inputs_set(ctx_, ioNum_.n_input, rknnInputs_);
    if (ret != RKNN_SUCC)
    {
        ROS_ERROR_THROTTLE(1.0, "rknn_inputs_set failed, ret=%d", ret);
        return;
    }
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC)
    {
        ROS_ERROR_THROTTLE(1.0, "rknn_run failed, ret=%d", ret);
        return;
    }
    ret = rknn_outputs_get(ctx_, ioNum_.n_output, rknnOutputs_, nullptr);
    if (ret != RKNN_SUCC)
    {
        ROS_ERROR_THROTTLE(1.0, "rknn_outputs_get failed, ret=%d", ret);
        return;
    }
    float* outputData = static_cast<float*>(rknnOutputs_[0].buf);
    for (int i = 0; i < numActions_; ++i)
    {
        float clampedValue = std::clamp(outputData[i], clipActionsLower_[i], clipActionsUpper_[i]);
        action_[i] = clampedValue;
    }
    rknn_outputs_release(ctx_, ioNum_.n_output, rknnOutputs_);
#endif
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
    if (msg->position.size() != static_cast<size_t>(numActions_))
    {
        ROS_WARN_THROTTLE(1.0, "Joint state size mismatch: %zu != %d", msg->position.size(), numActions_);
        return;
    }
    for (int i = 0; i < numActions_; ++i)
    {
        jointPositions_[i] = msg->position[i];
        if (msg->velocity.size() == msg->position.size())
        {
            jointVelocities_[i] = msg->velocity[i];
        }
    }
    if (!stateReceived_)
    {
        stateReceived_ = true;
        ROS_INFO("First robot state received.");
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
        ROS_INFO("First IMU data received.");
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
    ROS_INFO("Starting inference loop at %.1f Hz", rlCtrlFreq_);
    while (ros::ok() && !quit_)
    {
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
        rate.sleep();
    }
    ROS_INFO("Inference loop stopped.");
}

} // namespace inference_demo

