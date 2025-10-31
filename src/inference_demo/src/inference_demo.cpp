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
    ROS_INFO("RKNN input tensor configured, size: %d", obsInput_.size() * sizeof(float));

    memset(rknnOutputs_, 0, sizeof(rknnOutputs_));
    rknnOutputs_[0].want_float = true;
    ROS_INFO("RKNN output tensor configured");
    ROS_INFO("=== loadPolicy() Completed Successfully ===");
    return true;
#endif
}

void InferenceDemo::updateObservation(bool standby)
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
    static double time_counter = 0.0;
    if (standby) {
        // standby 模式：固定值
        observations_[10] = 1.0;
        observations_[11] = -1.0;
    } else {
        // running 模式：周期性 sin/cos
        time_counter += (1.0 / rlCtrlFreq_);
        observations_[10] = std::sin(2.0 * M_PI * time_counter * rlCtrlFreq_);
        observations_[11] = std::cos(2.0 * M_PI * time_counter * rlCtrlFreq_);
    }
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
    
    enum State { NOT_READY, STANDBY, RUNNING };
    static State currentState = NOT_READY;
    const char* stateNames[] = {"NOT_READY", "STANDBY", "RUNNING"};
    ROS_INFO("Initial state: %s (press LT+RT+START to reset and enter STANDBY)", stateNames[currentState]);
    
    int loopCount = 0;
    static ros::Time last_trigger(0);
    
    while (ros::ok() && !quit_)
    {
        ROS_DEBUG("=== Loop iteration %d, state=%s ===", loopCount++, stateNames[currentState]);
        ros::spinOnce();
        
        if (currentState == RUNNING) {
            updateObservation(false);  // RUNNING 模式：standby=false
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
                    stateNames[currentState],
                    (axis2 >= 0 && axis2 < (int)g_joy_msg.axes.size()) ? g_joy_msg.axes[axis2] : -999.0,
                    (axis5 >= 0 && axis5 < (int)g_joy_msg.axes.size()) ? g_joy_msg.axes[axis5] : -999.0,
                    start_pressed, lb_pressed, trigger_reset, trigger_toggle);
            }

            if (trigger_reset && (ros::Time::now() - last_trigger).toSec() > 1.0) {
                last_trigger = ros::Time::now();
                
                double reset_duration = 2.0;
                nh_->param("reset_duration", reset_duration, reset_duration);
                
                sensor_msgs::JointState preset;
                preset.header.frame_id = "zero";
                preset.header.stamp.fromSec(reset_duration);
                
                presetPub_.publish(preset);
                ROS_INFO("=== LT+RT+START: Reset to ZERO, transitioning to STANDBY ===");
                
                ros::Duration(reset_duration).sleep();
                currentState = STANDBY;
                ROS_INFO("=== State changed: %s (press LT+RT+LB to toggle RUNNING) ===", stateNames[currentState]);
            }
            
            if (trigger_toggle && (ros::Time::now() - last_trigger).toSec() > 1.0) {
                last_trigger = ros::Time::now();
                
                if (currentState == STANDBY) {
                    currentState = RUNNING;
                    ROS_INFO("=== LT+RT+LB: State changed to %s (policy inference enabled) ===", stateNames[currentState]);
                } else if (currentState == RUNNING) {
                    currentState = STANDBY;
                    ROS_INFO("=== LT+RT+LB: State changed to %s (policy inference disabled, legs send zero) ===", stateNames[currentState]);
                }
            }
        } else {
            static int no_joy_count = 0;
            if (++no_joy_count % 100 == 0) {
                ROS_WARN("Joystick not ready! State=%s", stateNames[currentState]);
            }
        }

        if (currentState == NOT_READY) {
            rate.sleep();
            continue;
        }
 
        // ========== 准备22维关节指令 ==========
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time(0);
        msg.position.resize(22);

        // ========== 1. 腿部12个关节（按策略输出顺序：0-11）==========
        // 极性（方向系数）：左腿6个 + 右腿6个
        const int leg_direction[12] = {
             1,  1, -1, -1,  1,  1,    // 左腿：hip_pitch, hip_roll, thigh, calf, ankle_pitch, ankle_roll
            -1,  1, -1,  1, -1,  1     // 右腿：hip_pitch, hip_roll, thigh, calf, ankle_pitch, ankle_roll
        };
        
        // 零位偏置（URDF offset）：左腿6个 + 右腿6个
        const double leg_offset[12] = {
            -0.25, 0.00, 0.00, 0.65, -0.40, 0.00,  // 左腿
            -0.25, 0.00, 0.00, 0.65, -0.40, 0.00   // 右腿
        };

        // 计算腿部输出：(基础值 * 极性) + offset
        // STANDBY: 基础值=0，RUNNING: 基础值=策略输出
        for (int i = 0; i < 12; ++i) {
            double base_value = (currentState == RUNNING) ? action_[i] : 0.0;
            msg.position[i] = (base_value + leg_offset[i]) * leg_direction[i];
        }

        // ========== 2. 左臂4个关节（索引：12-15）==========
        static std::vector<double> left_arm_cmd = {1.95, -1.57, 1.57, -1.57};
        nh_->param<std::vector<double>>("left_arm_cmd", left_arm_cmd, left_arm_cmd);
        
        const int left_arm_direction[4] = {1, -1, -1, -1};
        for (int i = 0; i < 4; ++i) {
            double cmd = (i < (int)left_arm_cmd.size() ? left_arm_cmd[i] : 0.0);
            msg.position[12 + i] = cmd * left_arm_direction[i];
        }

        // ========== 3. 右臂4个关节（索引：16-19）==========
        static std::vector<double> right_arm_cmd = {1.95, 1.57, -1.57, -1.57};
        nh_->param<std::vector<double>>("right_arm_cmd", right_arm_cmd, right_arm_cmd);
        
        const int right_arm_direction[4] = {-1, -1, -1, 1};
        for (int i = 0; i < 4; ++i) {
            double cmd = (i < (int)right_arm_cmd.size() ? right_arm_cmd[i] : 0.0);
            msg.position[16 + i] = cmd * right_arm_direction[i];
        }

        // ========== 4. 头部2个关节（索引：20-21）==========
        static std::vector<double> head_cmd(2, 0.0);
        nh_->param<std::vector<double>>("head_cmd", head_cmd, head_cmd);
        
        for (int i = 0; i < 2; ++i) {
            msg.position[20 + i] = (i < (int)head_cmd.size() ? head_cmd[i] : 0.0);
        }

        jointCmdPub_.publish(msg);
        
        static int log_count = 0;
        if (++log_count % 50 == 0) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(3);
            oss << "State=" << stateNames[currentState] << " | pub /" << modelType_ << "_all [";
            for (int i = 0; i < 22; ++i) {
                if (i) oss << ", ";
                oss << msg.position[i];
            }
            oss << "]";
            ROS_INFO("%s", oss.str().c_str());
        }
        
        rate.sleep();
    }
    ROS_INFO("=== Inference loop stopped ===");
}

} // namespace inference_demo

