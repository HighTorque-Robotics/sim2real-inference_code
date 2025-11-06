# HighTorque RL Custom Inference Demo
# 高擎机电强化学习推理演示

[English](#english) | [中文](#中文)

---

## English

### Overview

This is an open-source ROS1-based reinforcement learning inference demonstration package for HighTorque humanoid robots. It provides a complete example of how to deploy and run RL policies on real hardware using RKNN inference engine (Rockchip Neural Network).

**Developed by 高擎机电 (HighTorque Robotics)**

**Key Features:**
- 🤖 Real-time RL policy inference on ARM-based controllers
- 🔧 Easy-to-configure YAML parameter system
- 🎮 Joystick control for state transitions
- 📊 Comprehensive observation and action processing
- 🚀 100Hz control loop for smooth robot motion

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Control Layer                        │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│  │ /cmd_vel │         │   /joy   │         │ /imu/data│    │
│  └────┬─────┘         └────┬─────┘         └────┬─────┘    │
└───────┼────────────────────┼────────────────────┼──────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│          hightorque_rl_inference_node (This Package)         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Observation Processing (36-dim)                        │ │
│  │  • Gait phase (sin/cos)                                 │ │
│  │  • Command velocities (x, y, yaw)                       │ │
│  │  • Joint positions & velocities (12 DOF)                │ │
│  │  • Base angular velocity & orientation                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         RKNN Inference Engine (.rknn model)             │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Action Processing (12 DOF)                             │ │
│  │  • Action clipping & scaling                            │ │
│  │  • Motor direction mapping                              │ │
│  │  • State-based scaling (STANDBY/RUNNING)                │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Robot Control Layer                       │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ /pi_plus_all     │         │ /pi_plus_preset  │         │
│  │ (Joint Commands) │         │ (Reset Commands) │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  Robot Hardware  │
                   └──────────────────┘
```

### Prerequisites

**Hardware Requirements:**
- HighTorque humanoid robot (Pi Plus or compatible)
- ARM-based controller with RKNN support (e.g., RK3588)
- Joystick controller (for mode switching)

**Software Requirements:**
- Ubuntu 20.04 (or compatible)
- ROS1 Noetic
- Eigen3
- yaml-cpp
- RKNN runtime library (included in `lib/`)

### Installation

1. **Create a catkin workspace** (if you don't have one):
```bash
mkdir -p ~/catkin_ws
cd ~/catkin_ws
```

2. **Clone this repository**:
```bash
git clone https://github.com/HighTorque-Robotics/sim2real-inference_code.git
```

3. **Install dependencies**:
```bash
sudo apt-get update
sudo apt-get install ros-noetic-sensor-msgs ros-noetic-geometry-msgs \
                     libeigen3-dev libyaml-cpp-dev
```

4. **Build the package**:
```bash
cd ~/catkin_ws/sim2real-inference_code/
catkin init
catkin build
```

5. **Source the workspace**:
```bash
source devel/setup.bash
```

### Quick Start

#### Step 1: Start the Robot in Developer Mode

First, ensure your robot is running and in developer mode. This should start the following ROS topics:
- `/sim2real_master_node/rbt_state` - Robot joint states
- `/sim2real_master_node/mtr_state` - Motor states
- `/imu/data` - IMU data

#### Step 2: Configure Parameters

Edit the configuration file to match your robot and policy:
```bash
cd ~/catkin_ws/sim2real-inference_code/
nano config_example.yaml
```

Key parameters to configure:
- `policy_name`: Your RKNN model filename
- `num_actions`: Number of actuated joints (default: 12)
- `clip_actions_lower/upper`: Joint angle limits for your robot
- `motor_direction`: Motor rotation directions
- `map_index`: Joint order mapping

#### Step 3: Launch the Inference Node

```bash
roslaunch hightorque_rl_inference hightorque_rl_inference.launch
```

You should see output indicating:
```
[ INFO] Loading config from: /path/to/config_example.yaml
[ INFO] YAML config loaded successfully
[ INFO] Initialization successful, starting run loop
```

#### Step 4: Control the Robot

The system uses a **state machine** with three states:

1. **NOT_READY** (Initial State)
   - Robot is waiting for initialization
   - **Transition to STANDBY**: Press `LT + RT + START` on joystick

2. **STANDBY** (Ready State)
   - Robot is balanced but uses minimal action scale (0.05)
   - Safe mode for testing
   - **Transition to RUNNING**: Press `LT + RT + LB` on joystick

3. **RUNNING** (Active State)
   - Full RL policy execution with configured `action_scale`
   - Robot responds to `/cmd_vel` commands
   - **Transition to STANDBY**: Press `LT + RT + LB` again

**Sending velocity commands**:
```bash
# Move forward
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.5, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.0}"

# Turn left
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.0, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.5}"

# Stop
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.0, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.0}"
```

### Configuration Guide

See [docs/configuration.md](docs/configuration.md) for detailed parameter descriptions.

### ROS Topics

**Subscribed Topics:**
- `/sim2real_master_node/rbt_state` (sensor_msgs/JointState) - Robot joint positions and velocities
- `/sim2real_master_node/mtr_state` (sensor_msgs/JointState) - Motor absolute positions
- `/imu/data` (sensor_msgs/Imu) - IMU orientation and angular velocity
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
- `/joy` (sensor_msgs/Joy) - Joystick input

**Published Topics:**
- `/pi_plus_all` (sensor_msgs/JointState) - Joint position commands
- `/pi_plus_preset` (sensor_msgs/JointState) - Reset commands

### Troubleshooting

**Q: "Timeout waiting for robot data"**
- Ensure the robot is running and topics are being published
- Check topic names with `rostopic list`
- Verify topic data with `rostopic echo /sim2real_master_node/rbt_state`

**Q: "Model loading failed"**
- Check that the `.rknn` model file exists in `policy/` directory
- Verify `policy_name` in `config_example.yaml` matches your file
- Ensure RKNN runtime library is correctly installed

**Q: Robot behaves erratically**
- Check `motor_direction` configuration
- Verify `map_index` matches your robot's joint ordering
- Adjust `action_scale` to a lower value
- Review `clip_actions_lower/upper` limits

For more issues, see [docs/troubleshooting.md](docs/troubleshooting.md)

### Development Guide

#### Adding Your Own RL Policy

1. Convert your trained policy to RKNN format (`.rknn` file)
2. Place it in the `policy/` directory
3. Update `config_example.yaml` with:
   - New `policy_name`
   - Correct `num_single_obs` and `num_actions`
   - Appropriate scaling parameters
4. Test in STANDBY mode first before switching to RUNNING

#### Modifying Observation Space

Edit `src/hightorque_rl_inference.cpp`, function `updateObservation()`:
```cpp
void InferenceDemo::updateObservation()
{
    // Resize observations if needed
    observations_.resize(numSingleObs_);
    
    // Add your custom observations
    observations_[0] = /* your observation 1 */;
    observations_[1] = /* your observation 2 */;
    // ...
}
```

See [docs/development.md](docs/development.md) for more details.

### Project Structure

```
hightorque_rl_custom/
├── src/
│   └── hightorque_rl_inference/
│       ├── CMakeLists.txt          # Build configuration
│       ├── package.xml             # Package metadata
│       ├── config_example.yaml     # Default configuration
│       ├── include/
│       │   ├── hightorque_rl_inference/
│       │   │   └── hightorque_rl_inference.h    # Main class header
│       │   └── rknn/
│       │       └── rknn_api.h          # RKNN API header
│       ├── launch/
│       │   └── hightorque_rl_inference.launch   # Launch file
│       ├── lib/
│       │   └── librknnrt.so            # RKNN runtime library
│       ├── policy/
│       │   ├── policy_0322_12dof_4000.rknn  # Example model
│       │   └── combined_model_dwaq_v1226.rknn
│       └── src/
│           ├── hightorque_rl_inference.cpp      # Main implementation
│           └── main.cpp                # Entry point
├── docs/                           # Documentation
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 中文

### 项目简介

这是一个基于 ROS1 的开源强化学习推理演示包，专为 HighTorque 人形机器人设计。它提供了一个完整的示例，展示如何使用 RKNN 推理引擎（Rockchip Neural Network）在真实硬件上部署和运行强化学习策略。

**开发商：高擎机电（HighTorque Robotics）**

**核心特性：**
- 🤖 在 ARM 架构控制器上实时运行强化学习策略推理
- 🔧 简单易用的 YAML 参数配置系统
- 🎮 手柄控制状态切换
- 📊 完整的观测值和动作处理流程
- 🚀 100Hz 控制频率，实现流畅的机器人运动

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户控制层                              │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│  │ /cmd_vel │         │   /joy   │         │ /imu/data│    │
│  └────┬─────┘         └────┬─────┘         └────┬─────┘    │
└───────┼────────────────────┼────────────────────┼──────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│        hightorque_rl_inference_node (本功能包)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  观测值处理 (36维)                                       │ │
│  │  • 步态相位 (sin/cos)                                    │ │
│  │  • 速度指令 (x, y, yaw)                                  │ │
│  │  • 关节位置和速度 (12自由度)                            │ │
│  │  • 基座角速度和姿态                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         RKNN 推理引擎 (.rknn 模型)                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  动作处理 (12自由度)                                     │ │
│  │  • 动作裁剪和缩放                                        │ │
│  │  • 电机方向映射                                          │ │
│  │  • 基于状态的缩放 (STANDBY/RUNNING)                     │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      机器人控制层                            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ /pi_plus_all     │         │ /pi_plus_preset  │         │
│  │ (关节指令)       │         │ (复位指令)       │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │    机器人硬件    │
                   └──────────────────┘
```

### 环境要求

**硬件要求：**
- HighTorque 人形机器人（Pi Plus 或兼容机型）
- 支持 RKNN 的 ARM 控制器（如 RK3588）
- 游戏手柄控制器（用于模式切换）

**软件要求：**
- Ubuntu 20.04（或兼容版本）
- ROS1 Noetic
- Eigen3
- yaml-cpp
- RKNN 运行时库（已包含在 `lib/` 目录）

### 安装步骤

1. **创建 catkin 工作空间**（如果还没有）：
```bash
mkdir -p ~/catkin_ws
cd ~/catkin_ws
```

2. **克隆本仓库**：
```bash
git clone https://github.com/HighTorque-Robotics/sim2real-inference_code.git
```

3. **安装依赖**：
```bash
sudo apt-get update
sudo apt-get install ros-noetic-sensor-msgs ros-noetic-geometry-msgs \
                     libeigen3-dev libyaml-cpp-dev
```

4. **编译功能包**：
```bash
cd ~/catkin_ws/sim2real-inference_code/
catkin init
catkin build
```

5. **加载工作空间环境**：
```bash
source devel/setup.bash
```

### 快速开始

#### 步骤 1：启动机器人开发者模式

首先，确保你的机器人正在运行并处于开发者模式。这将启动以下 ROS 话题：
- `/sim2real_master_node/rbt_state` - 机器人关节状态
- `/sim2real_master_node/mtr_state` - 电机状态
- `/imu/data` - IMU 数据

#### 步骤 2：配置参数

编辑配置文件以匹配你的机器人和策略：
```bash
cd ~/catkin_ws/sim2real-inference_code/
nano config_example.yaml
```

需要配置的关键参数：
- `policy_name`: 你的 RKNN 模型文件名
- `num_actions`: 驱动关节数量（默认：12）
- `clip_actions_lower/upper`: 机器人的关节角度限制
- `motor_direction`: 电机旋转方向
- `map_index`: 关节顺序映射

#### 步骤 3：启动推理节点

```bash
roslaunch hightorque_rl_inference hightorque_rl_inference.launch
```

你应该看到以下输出：
```
[ INFO] Loading config from: /path/to/config_example.yaml
[ INFO] YAML config loaded successfully
[ INFO] Initialization successful, starting run loop
```

#### 步骤 4：控制机器人

系统使用**状态机**，包含三个状态：

1. **NOT_READY（未就绪）**（初始状态）
   - 机器人等待初始化
   - **切换到 STANDBY**：按下手柄上的 `LT + RT + START`

2. **STANDBY（待机）**（就绪状态）
   - 机器人保持平衡但使用最小动作缩放（0.05）
   - 测试安全模式
   - **切换到 RUNNING**：按下手柄上的 `LT + RT + LB`

3. **RUNNING（运行）**（活动状态）
   - 使用配置的 `action_scale` 完整执行强化学习策略
   - 机器人响应 `/cmd_vel` 指令
   - **切换回 STANDBY**：再次按下 `LT + RT + LB`

**发送速度指令**：
```bash
# 前进
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.5, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.0}"

# 左转
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.0, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.5}"

# 停止
rostopic pub /cmd_vel geometry_msgs/Twist \
  "linear: {x: 0.0, y: 0.0, z: 0.0}
   angular: {x: 0.0, y: 0.0, z: 0.0}"
```

### 配置指南

详细的参数说明请参见 [docs/configuration.md](docs/configuration.md)

### ROS 话题

**订阅的话题：**
- `/sim2real_master_node/rbt_state` (sensor_msgs/JointState) - 机器人关节位置和速度
- `/sim2real_master_node/mtr_state` (sensor_msgs/JointState) - 电机绝对位置
- `/imu/data` (sensor_msgs/Imu) - IMU 姿态和角速度
- `/cmd_vel` (geometry_msgs/Twist) - 速度指令
- `/joy` (sensor_msgs/Joy) - 手柄输入

**发布的话题：**
- `/pi_plus_all` (sensor_msgs/JointState) - 关节位置指令
- `/pi_plus_preset` (sensor_msgs/JointState) - 复位指令

### 常见问题

**问："Timeout waiting for robot data"**
- 确保机器人正在运行且话题正在发布
- 使用 `rostopic list` 检查话题名称
- 使用 `rostopic echo /sim2real_master_node/rbt_state` 验证话题数据

**问："Model loading failed"**
- 检查 `.rknn` 模型文件是否存在于 `policy/` 目录
- 验证 `config_example.yaml` 中的 `policy_name` 与文件名匹配
- 确保 RKNN 运行时库已正确安装

**问：机器人行为异常**
- 检查 `motor_direction` 配置
- 验证 `map_index` 与机器人的关节顺序匹配
- 将 `action_scale` 调整为较小的值
- 检查 `clip_actions_lower/upper` 限制

更多问题请参见 [docs/troubleshooting.md](docs/troubleshooting.md)

### 开发指南

#### 添加自己的强化学习策略

1. 将训练好的策略转换为 RKNN 格式（`.rknn` 文件）
2. 将其放置在 `policy/` 目录
3. 更新 `config_example.yaml`：
   - 新的 `policy_name`
   - 正确的 `num_single_obs` 和 `num_actions`
   - 适当的缩放参数
4. 先在 STANDBY 模式下测试，然后再切换到 RUNNING

#### 修改观测空间

编辑 `src/hightorque_rl_inference.cpp`，修改 `updateObservation()` 函数：
```cpp
void InferenceDemo::updateObservation()
{
    // 如需要，调整观测维度
    observations_.resize(numSingleObs_);
    
    // 添加自定义观测
    observations_[0] = /* 你的观测值 1 */;
    observations_[1] = /* 你的观测值 2 */;
    // ...
}
```

更多详情请参见 [docs/development.md](docs/development.md)

### 项目结构

```
hightorque_rl_custom/
├── src/
│   └── hightorque_rl_inference/
│       ├── CMakeLists.txt          # 编译配置
│       ├── package.xml             # 功能包元数据
│       ├── config_example.yaml     # 默认配置
│       ├── include/
│       │   ├── hightorque_rl_inference/
│       │   │   └── hightorque_rl_inference.h    # 主类头文件
│       │   └── rknn/
│       │       └── rknn_api.h          # RKNN API 头文件
│       ├── launch/
│       │   └── hightorque_rl_inference.launch   # 启动文件
│       ├── lib/
│       │   └── librknnrt.so            # RKNN 运行时库
│       ├── policy/
│       │   ├── policy_0322_12dof_4000.rknn  # 示例模型
│       │   └── combined_model_dwaq_v1226.rknn
│       └── src/
│           ├── hightorque_rl_inference.cpp      # 主实现
│           └── main.cpp                # 程序入口
├── docs/                           # 文档目录
├── README.md                       # 本文件
└── .gitignore                      # Git 忽略规则
```