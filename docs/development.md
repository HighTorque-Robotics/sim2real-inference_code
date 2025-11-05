# Development Guide / 开发指南

**Developed by 高擎机电 (HighTorque Robotics)**

[English](#english) | [中文](#中文)

---

## English

This guide explains how to extend and modify the HighTorque RL Custom Inference Demo for your own robotics projects.

### Project Structure

```
src/hightorque_rl_inference/
├── CMakeLists.txt          # Build configuration
├── package.xml             # Package metadata
├── config_example.yaml     # Runtime configuration
├── include/
│   ├── hightorque_rl_inference/
│   │   └── hightorque_rl_inference.h    # Main class declaration
│   └── rknn/
│       └── rknn_api.h          # RKNN API headers
├── launch/
│   └── hightorque_rl_inference.launch   # Launch file
├── lib/
│   └── librknnrt.so            # RKNN runtime library
├── policy/
│   └── *.rknn                  # RKNN model files
└── src/
    ├── hightorque_rl_inference.cpp      # Main implementation
    └── main.cpp                # Entry point
```

### Key Classes and Functions

#### InferenceDemo Class

The main class handling RL inference:

**Key Members:**
- `numActions_`: Number of robot joints (DOF)
- `numSingleObs_`: Observation vector dimension
- `observations_`: Current observation vector
- `action_`: Current action vector
- `ctx_`: RKNN inference context

**Key Methods:**
- `init()`: Initialize ROS topics and load policy
- `loadPolicy()`: Load RKNN model from file
- `updateObservation()`: Build observation vector
- `updateAction()`: Run inference to generate actions
- `run()`: Main control loop

---

## Customizing Observation Space

### Current Observation Structure (36-dim)

```cpp
observations_[0]     = sin(2*pi*step)      // Gait phase
observations_[1]     = cos(2*pi*step)      // Gait phase
observations_[2:4]   = cmd velocities      // x, y, yaw
observations_[5:16]  = joint positions     // 12 DOF
observations_[17:28] = joint velocities    // 12 DOF
observations_[29:31] = base angular vel    // 3-axis
observations_[32:34] = base orientation    // Euler angles
```

### Adding New Observations

**Example: Add base linear velocity**

1. **Increase `num_single_obs` in config:**
   ```yaml
   num_single_obs: 39  # Was 36, now 36+3
   ```

2. **Subscribe to additional topic** in `InferenceDemo` constructor:
   ```cpp
   baseVelSub_ = nh_->subscribe("/base_velocity", 10, 
       &InferenceDemo::baseVelCallback, this);
   ```

3. **Add callback to store data:**
   ```cpp
   void InferenceDemo::baseVelCallback(const geometry_msgs::TwistConstPtr& msg)
   {
       baseLinVel_(0) = msg->linear.x;
       baseLinVel_(1) = msg->linear.y;
       baseLinVel_(2) = msg->linear.z;
   }
   ```

4. **Update `updateObservation()` function:**
   ```cpp
   void InferenceDemo::updateObservation()
   {
       // ... existing observations ...
       
       // Add new observations at the end
       observations_[35] = baseLinVel_(0);
       observations_[36] = baseLinVel_(1);
       observations_[37] = baseLinVel_(2);
       
       // ... rest of function ...
   }
   ```

5. **Retrain your policy** with the new observation space

---

## Customizing Action Space

### Current Action Structure (12-dim)

12 joint position targets, one per actuated joint.

### Adding New Actions

**Example: Include gripper control**

1. **Increase `num_actions` in config:**
   ```yaml
   num_actions: 13  # Was 12, now 12+1 for gripper
   ```

2. **Update action limits:**
   ```yaml
   clip_actions_lower: [-1.00, ..., 0.0]   # Add gripper lower
   clip_actions_upper: [1.00, ..., 1.0]    # Add gripper upper
   ```

3. **Modify `run()` function** to handle gripper action:
   ```cpp
   void InferenceDemo::run()
   {
       // ... existing code ...
       
       // Separate gripper action
       double gripperCmd = action_[12];
       
       // Publish gripper command
       std_msgs::Float64 gripperMsg;
       gripperMsg.data = gripperCmd;
       gripperPub_.publish(gripperMsg);
   }
   ```

---

## Model Conversion

### PyTorch → ONNX → RKNN

#### Step 1: Export to ONNX

```python
import torch
import torch.onnx

# Load your trained policy
policy = torch.jit.load("policy.pt")
policy.eval()

# Create dummy input (batch_size=1, obs_dim=36)
dummy_input = torch.randn(1, 36)

# Export to ONNX
torch.onnx.export(
    policy,
    dummy_input,
    "policy.onnx",
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={
        'observation': {0: 'batch_size'},
        'action': {0: 'batch_size'}
    },
    opset_version=11
)
```

#### Step 2: Convert ONNX to RKNN

```python
from rknn.api import RKNN

# Create RKNN object
rknn = RKNN(verbose=True)

# Config
print('--> Config model')
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    target_platform='rk3588'
)

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(model='policy.onnx')

# Build RKNN model
print('--> Building model')
ret = rknn.build(do_quantization=False)

# Export RKNN model
print('--> Export model')
ret = rknn.export_rknn('policy.rknn')

rknn.release()
```

#### Step 3: Quantization (Optional, for better performance)

```python
# Create dataset for quantization
def dataset_generator():
    # Generate or load representative observation samples
    for i in range(100):
        obs = np.random.randn(1, 36).astype(np.float32)
        yield [obs]

# Build with quantization
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8'
)

rknn.build(do_quantization=True, dataset=dataset_generator)
rknn.export_rknn('policy_quantized.rknn')
```

---

## Debugging Techniques

### 1. Print Observation Values

Add to `updateObservation()`:
```cpp
void InferenceDemo::updateObservation()
{
    // ... build observations ...
    
    // Debug print
    std::cout << "Obs: ";
    for (int i = 0; i < numSingleObs_; ++i) {
        std::cout << observations_[i] << " ";
    }
    std::cout << std::endl;
}
```

### 2. Log Action Values

Add to `updateAction()`:
```cpp
void InferenceDemo::updateAction()
{
    // ... run inference ...
    
    // Debug print
    ROS_INFO_STREAM("Actions: " << action_.transpose());
}
```

### 3. Visualize Observations in RViz

Publish observations as a custom message:
```cpp
// In run() function
std_msgs::Float64MultiArray obsMsg;
obsMsg.data.assign(observations_.data(), 
                   observations_.data() + observations_.size());
obsPub_.publish(obsMsg);
```

### 4. Record Data for Analysis

```bash
# Record all relevant topics
rosbag record /sim2real_master_node/rbt_state \
              /sim2real_master_node/mtr_state \
              /imu/data \
              /cmd_vel \
              /pi_plus_all \
              -O test_run.bag

# Play back later
rosbag play test_run.bag
```

---

## Testing Best Practices

### 1. Unit Testing

Create test files in `test/` directory:

```cpp
// test/test_observation.cpp
#include <gtest/gtest.h>
#include "hightorque_rl_inference/hightorque_rl_inference.h"

TEST(InferenceDemoTest, ObservationSize)
{
    // Test observation vector has correct size
    auto nh = std::make_shared<ros::NodeHandle>();
    InferenceDemo demo(nh);
    
    EXPECT_EQ(demo.getObservationSize(), 36);
}

// Add to CMakeLists.txt
catkin_add_gtest(${PROJECT_NAME}_test test/test_observation.cpp)
```

### 2. Integration Testing

Test on robot hardware:
1. Start in STANDBY mode
2. Verify all joints receive commands
3. Check for unexpected oscillations
4. Test state transitions
5. Test emergency stop

### 3. Performance Profiling

```cpp
#include <chrono>

void InferenceDemo::updateAction()
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // ... inference code ...
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    ROS_INFO("Inference time: %ld μs", duration.count());
}
```

---

## Contributing

### Code Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use meaningful variable names
- Add Doxygen comments for all functions
- Keep functions < 50 lines when possible

### Commit Messages

```
[Component] Brief description

Detailed explanation of what changed and why.

Fixes #issue_number
```

Example:
```
[Observation] Add base linear velocity to observation space

- Added baseVelSub_ subscriber
- Updated updateObservation() to include linear velocity
- Increased num_single_obs to 39

Fixes #42
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test thoroughly
4. Update documentation
5. Commit with clear messages
6. Push to your fork
7. Open pull request with description

---

## 中文

本指南解释如何扩展和修改 HighTorque RL 自定义推理演示以用于您自己的机器人项目。

### 项目结构

```
src/hightorque_rl_inference/
├── CMakeLists.txt          # 构建配置
├── package.xml             # 包元数据
├── config_example.yaml     # 运行时配置
├── include/
│   ├── hightorque_rl_inference/
│   │   └── hightorque_rl_inference.h    # 主类声明
│   └── rknn/
│       └── rknn_api.h          # RKNN API 头文件
├── launch/
│   └── hightorque_rl_inference.launch   # 启动文件
├── lib/
│   └── librknnrt.so            # RKNN 运行时库
├── policy/
│   └── *.rknn                  # RKNN 模型文件
└── src/
    ├── hightorque_rl_inference.cpp      # 主实现
    └── main.cpp                # 入口点
```

### 关键类和函数

#### InferenceDemo 类

处理 RL 推理的主类：

**关键成员：**
- `numActions_`：机器人关节数（自由度）
- `numSingleObs_`：观测向量维度
- `observations_`：当前观测向量
- `action_`：当前动作向量
- `ctx_`：RKNN 推理上下文

**关键方法：**
- `init()`：初始化 ROS 话题并加载策略
- `loadPolicy()`：从文件加载 RKNN 模型
- `updateObservation()`：构建观测向量
- `updateAction()`：运行推理以生成动作
- `run()`：主控制循环

---

## 自定义观测空间

### 当前观测结构（36维）

```cpp
observations_[0]     = sin(2*pi*step)      // 步态相位
observations_[1]     = cos(2*pi*step)      // 步态相位
observations_[2:4]   = cmd velocities      // x, y, yaw
observations_[5:16]  = joint positions     // 12 自由度
observations_[17:28] = joint velocities    // 12 自由度
observations_[29:31] = base angular vel    // 3 轴
observations_[32:34] = base orientation    // 欧拉角
```

### 添加新观测

**示例：添加基座线速度**

1. **在配置中增加 `num_single_obs`：**
   ```yaml
   num_single_obs: 39  # 原来是 36，现在 36+3
   ```

2. **在 `InferenceDemo` 构造函数中订阅额外话题：**
   ```cpp
   baseVelSub_ = nh_->subscribe("/base_velocity", 10, 
       &InferenceDemo::baseVelCallback, this);
   ```

3. **添加回调以存储数据：**
   ```cpp
   void InferenceDemo::baseVelCallback(const geometry_msgs::TwistConstPtr& msg)
   {
       baseLinVel_(0) = msg->linear.x;
       baseLinVel_(1) = msg->linear.y;
       baseLinVel_(2) = msg->linear.z;
   }
   ```

4. **更新 `updateObservation()` 函数：**
   ```cpp
   void InferenceDemo::updateObservation()
   {
       // ... 现有观测 ...
       
       // 在末尾添加新观测
       observations_[35] = baseLinVel_(0);
       observations_[36] = baseLinVel_(1);
       observations_[37] = baseLinVel_(2);
       
       // ... 函数的其余部分 ...
   }
   ```

5. **使用新观测空间重新训练策略**

---

## 自定义动作空间

### 当前动作结构（12维）

12 个关节位置目标，每个驱动关节一个。

### 添加新动作

**示例：包含夹爪控制**

1. **在配置中增加 `num_actions`：**
   ```yaml
   num_actions: 13  # 原来是 12，现在 12+1 用于夹爪
   ```

2. **更新动作限制：**
   ```yaml
   clip_actions_lower: [-1.00, ..., 0.0]   # 添加夹爪下限
   clip_actions_upper: [1.00, ..., 1.0]    # 添加夹爪上限
   ```

3. **修改 `run()` 函数**以处理夹爪动作：
   ```cpp
   void InferenceDemo::run()
   {
       // ... 现有代码 ...
       
       // 分离夹爪动作
       double gripperCmd = action_[12];
       
       // 发布夹爪命令
       std_msgs::Float64 gripperMsg;
       gripperMsg.data = gripperCmd;
       gripperPub_.publish(gripperMsg);
   }
   ```

---

## 模型转换

### PyTorch → ONNX → RKNN

#### 步骤 1：导出到 ONNX

```python
import torch
import torch.onnx

# 加载训练好的策略
policy = torch.jit.load("policy.pt")
policy.eval()

# 创建虚拟输入 (batch_size=1, obs_dim=36)
dummy_input = torch.randn(1, 36)

# 导出到 ONNX
torch.onnx.export(
    policy,
    dummy_input,
    "policy.onnx",
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={
        'observation': {0: 'batch_size'},
        'action': {0: 'batch_size'}
    },
    opset_version=11
)
```

#### 步骤 2：将 ONNX 转换为 RKNN

```python
from rknn.api import RKNN

# 创建 RKNN 对象
rknn = RKNN(verbose=True)

# 配置
print('--> Config model')
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    target_platform='rk3588'
)

# 加载 ONNX 模型
print('--> Loading model')
ret = rknn.load_onnx(model='policy.onnx')

# 构建 RKNN 模型
print('--> Building model')
ret = rknn.build(do_quantization=False)

# 导出 RKNN 模型
print('--> Export model')
ret = rknn.export_rknn('policy.rknn')

rknn.release()
```

#### 步骤 3：量化（可选，以获得更好的性能）

```python
# 为量化创建数据集
def dataset_generator():
    # 生成或加载代表性观测样本
    for i in range(100):
        obs = np.random.randn(1, 36).astype(np.float32)
        yield [obs]

# 使用量化构建
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8'
)

rknn.build(do_quantization=True, dataset=dataset_generator)
rknn.export_rknn('policy_quantized.rknn')
```

---

## 调试技巧

### 1. 打印观测值

添加到 `updateObservation()`：
```cpp
void InferenceDemo::updateObservation()
{
    // ... 构建观测 ...
    
    // 调试打印
    std::cout << "Obs: ";
    for (int i = 0; i < numSingleObs_; ++i) {
        std::cout << observations_[i] << " ";
    }
    std::cout << std::endl;
}
```

### 2. 记录动作值

添加到 `updateAction()`：
```cpp
void InferenceDemo::updateAction()
{
    // ... 运行推理 ...
    
    // 调试打印
    ROS_INFO_STREAM("Actions: " << action_.transpose());
}
```

### 3. 在 RViz 中可视化观测

将观测发布为自定义消息：
```cpp
// 在 run() 函数中
std_msgs::Float64MultiArray obsMsg;
obsMsg.data.assign(observations_.data(), 
                   observations_.data() + observations_.size());
obsPub_.publish(obsMsg);
```

### 4. 记录数据以供分析

```bash
# 记录所有相关话题
rosbag record /sim2real_master_node/rbt_state \
              /sim2real_master_node/mtr_state \
              /imu/data \
              /cmd_vel \
              /pi_plus_all \
              -O test_run.bag

# 稍后回放
rosbag play test_run.bag
```

---

## 测试最佳实践

### 1. 单元测试

在 `test/` 目录中创建测试文件：

```cpp
// test/test_observation.cpp
#include <gtest/gtest.h>
#include "hightorque_rl_inference/hightorque_rl_inference.h"

TEST(InferenceDemoTest, ObservationSize)
{
    // 测试观测向量是否有正确的大小
    auto nh = std::make_shared<ros::NodeHandle>();
    InferenceDemo demo(nh);
    
    EXPECT_EQ(demo.getObservationSize(), 36);
}

// 添加到 CMakeLists.txt
catkin_add_gtest(${PROJECT_NAME}_test test/test_observation.cpp)
```

### 2. 集成测试

在机器人硬件上测试：
1. 在 STANDBY 模式下启动
2. 验证所有关节是否接收指令
3. 检查意外振荡
4. 测试状态转换
5. 测试紧急停止

### 3. 性能分析

```cpp
#include <chrono>

void InferenceDemo::updateAction()
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // ... 推理代码 ...
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    ROS_INFO("Inference time: %ld μs", duration.count());
}
```

---

## 贡献

### 代码风格

- 遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 使用有意义的变量名
- 为所有函数添加 Doxygen 注释
- 尽可能保持函数 < 50 行

### 提交消息

```
[组件] 简短描述

详细解释更改了什么以及为什么。

Fixes #issue_number
```

示例：
```
[Observation] 向观测空间添加基座线速度

- 添加了 baseVelSub_ 订阅者
- 更新了 updateObservation() 以包含线速度
- 将 num_single_obs 增加到 39

Fixes #42
```

### Pull Request 流程

1. Fork 仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 进行更改并彻底测试
4. 更新文档
5. 使用清晰的消息提交
6. 推送到您的 fork
7. 打开带有描述的 pull request

---

**高擎机电（HighTorque Robotics）**  
**最后更新**：2025年11月

