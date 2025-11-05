# Configuration Guide / 配置指南

**Developed by 高擎机电 (HighTorque Robotics)**

[English](#english) | [中文](#中文)

---

## English

This document provides detailed explanations of all configuration parameters in `config_example.yaml`.

### Basic Parameters

#### `num_actions`
- **Type**: Integer
- **Default**: 12
- **Description**: Number of actuated joints (DOF) on the robot
- **Usage**: Must match your robot's actual joint count and your trained policy output dimension

#### `num_single_obs`
- **Type**: Integer
- **Default**: 36
- **Description**: Dimension of observation vector for a single timestep
- **Composition**:
  - 2: Gait phase (sin/cos)
  - 3: Command velocities (x, y, yaw)
  - 12: Joint positions
  - 12: Joint velocities
  - 3: Base angular velocity
  - 3: Base orientation (Euler angles)
  - Total: 2 + 3 + 12 + 12 + 3 + 3 = 35 (typically padded to 36)

#### `frame_stack`
- **Type**: Integer
- **Default**: 1
- **Description**: Number of observation frames to stack as input to policy
- **Usage**: If > 1, provides temporal information. Input dimension = `num_single_obs * frame_stack`

### Model Configuration

#### `policy_name`
- **Type**: String
- **Default**: "policy_0322_12dof_4000.rknn"
- **Description**: Filename of the RKNN model (must be in `policy/` directory)
- **Format**: Must be `.rknn` format for ARM RKNN inference
- **Example**: "combined_model_dwaq_v1226.rknn"

### Control Frequency

#### `dt`
- **Type**: Float
- **Default**: 0.001
- **Description**: Simulation time step in seconds
- **Usage**: Used with `decimation` to calculate control frequency

#### `decimation`
- **Type**: Integer
- **Default**: 10
- **Description**: Number of simulation steps per control step
- **Usage**: Control frequency = 1 / (dt * decimation) = 1 / (0.001 * 10) = 100 Hz

### Gait Parameters

#### `frequency`
- **Type**: Float
- **Default**: 0.5
- **Description**: Base gait frequency in Hz
- **Usage**: Determines the period of the sinusoidal gait phase signal
- **Typical Range**: 0.3 - 1.0 Hz

### Observation Scaling

These parameters scale raw sensor readings before feeding to the policy:

#### `cmd_lin_vel_scale`
- **Type**: Float
- **Default**: 1.0
- **Description**: Linear velocity command scaling factor
- **Usage**: Higher values make the policy more sensitive to linear velocity commands

#### `cmd_ang_vel_scale`
- **Type**: Float
- **Default**: 1.25
- **Description**: Angular velocity command scaling factor
- **Usage**: Higher values make the policy more sensitive to turning commands

#### `rbt_lin_pos_scale`
- **Type**: Float
- **Default**: 1.0
- **Description**: Joint position scaling factor
- **Usage**: Should match training scaling

#### `rbt_lin_vel_scale`
- **Type**: Float
- **Default**: 1.0
- **Description**: Joint velocity scaling factor

#### `rbt_ang_vel_scale`
- **Type**: Float
- **Default**: 1.0
- **Description**: Base angular velocity scaling factor

#### `clip_obs`
- **Type**: Float
- **Default**: 18.0
- **Description**: Maximum absolute value for observation clipping
- **Usage**: Prevents extreme sensor values from destabilizing the policy

### Action Configuration

#### `action_scale`
- **Type**: Float
- **Default**: 1.0
- **Description**: Global scaling factor applied to all policy outputs
- **Tuning**: Start with lower values (0.5-0.7) for safety, increase gradually

#### `clip_actions_lower`
- **Type**: Array of 12 floats
- **Default**: `[-1.00, -0.40, -0.60, -1.30, -0.75, -0.30, -1.00, -0.40, -0.60, -1.30, -0.75, -0.30]`
- **Description**: Lower limits for each joint action (in radians)
- **Safety**: These limits prevent the robot from exceeding safe joint ranges

#### `clip_actions_upper`
- **Type**: Array of 12 floats
- **Default**: `[1.00, 0.40, 0.60, 1.30, 0.75, 0.30, 1.00, 0.40, 0.60, 1.30, 0.75, 0.30]`
- **Description**: Upper limits for each joint action (in radians)

### Motor Configuration

#### `motor_direction`
- **Type**: Array of 12 integers
- **Default**: `[1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1]`
- **Description**: Direction multiplier for each motor (1 or -1)
- **Usage**: Corrects for motor mounting orientation differences

#### `map_index`
- **Type**: Array of 12 integers
- **Default**: `[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]`
- **Description**: Maps policy joint order to actual motor order
- **Usage**: `actual_motor[i] = policy_joint[map_index[i]]`

#### `urdf_dof_pos_offset`
- **Type**: Array of 12 floats
- **Default**: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- **Description**: Offset between URDF zero position and actual motor zero position
- **Usage**: Compensates for calibration differences

---

## Tuning Tips

### For Walking Stability
1. **Start conservative**: Use `action_scale: 0.5` initially
2. **Test in STANDBY mode** first (uses `action_scale: 0.05`)
3. **Gradually increase** action_scale until desired performance
4. **Check joint limits**: Ensure `clip_actions_lower/upper` match physical limits

### For Velocity Control
1. **Adjust command scales**: Increase `cmd_lin_vel_scale` / `cmd_ang_vel_scale` for more responsive control
2. **Test incrementally**: Start with slow commands and increase

### For Different Robots
1. **Update `motor_direction`**: Match your robot's motor mounting
2. **Update `map_index`**: Match your robot's joint ordering
3. **Update action limits**: Match your robot's physical joint ranges
4. **Recalibrate `urdf_dof_pos_offset`**: If your robot uses different zero positions

---

## Example Configurations

### Conservative (Safe Testing)
```yaml
action_scale: 0.5
cmd_lin_vel_scale: 0.8
cmd_ang_vel_scale: 1.0
```

### Aggressive (Fast Response)
```yaml
action_scale: 1.2
cmd_lin_vel_scale: 1.5
cmd_ang_vel_scale: 1.5
```

### Multi-Policy Setup
You can maintain multiple configuration files:
```bash
config_conservative.yaml
config_normal.yaml
config_aggressive.yaml
```

Load different configs by modifying the launch file:
```xml
<param name="config_file" value="$(find hightorque_rl_inference)/config_aggressive.yaml"/>
```

---

## 中文

本文档提供 `config_example.yaml` 中所有配置参数的详细说明。

### 基本参数

#### `num_actions`
- **类型**：整数
- **默认值**：12
- **描述**：机器人驱动关节数量（自由度）
- **用法**：必须与机器人实际关节数和训练策略的输出维度匹配

#### `num_single_obs`
- **类型**：整数
- **默认值**：36
- **描述**：单个时间步的观测向量维度
- **组成**：
  - 2：步态相位（sin/cos）
  - 3：速度指令（x, y, yaw）
  - 12：关节位置
  - 12：关节速度
  - 3：基座角速度
  - 3：基座姿态（欧拉角）
  - 总计：2 + 3 + 12 + 12 + 3 + 3 = 35（通常填充到 36）

#### `frame_stack`
- **类型**：整数
- **默认值**：1
- **描述**：堆叠的观测帧数作为策略输入
- **用法**：如果 > 1，提供时间信息。输入维度 = `num_single_obs * frame_stack`

### 模型配置

#### `policy_name`
- **类型**：字符串
- **默认值**："policy_0322_12dof_4000.rknn"
- **描述**：RKNN 模型文件名（必须在 `policy/` 目录中）
- **格式**：必须是 ARM RKNN 推理的 `.rknn` 格式
- **示例**："combined_model_dwaq_v1226.rknn"

### 控制频率

#### `dt`
- **类型**：浮点数
- **默认值**：0.001
- **描述**：仿真时间步长（秒）
- **用法**：与 `decimation` 一起计算控制频率

#### `decimation`
- **类型**：整数
- **默认值**：10
- **描述**：每个控制步的仿真步数
- **用法**：控制频率 = 1 / (dt * decimation) = 1 / (0.001 * 10) = 100 Hz

### 步态参数

#### `frequency`
- **类型**：浮点数
- **默认值**：0.5
- **描述**：基本步态频率（Hz）
- **用法**：决定正弦步态相位信号的周期
- **典型范围**：0.3 - 1.0 Hz

### 观测缩放

这些参数在馈送到策略之前缩放原始传感器读数：

#### `cmd_lin_vel_scale`
- **类型**：浮点数
- **默认值**：1.0
- **描述**：线速度指令缩放因子
- **用法**：较高的值使策略对线速度指令更敏感

#### `cmd_ang_vel_scale`
- **类型**：浮点数
- **默认值**：1.25
- **描述**：角速度指令缩放因子
- **用法**：较高的值使策略对转向指令更敏感

#### `rbt_lin_pos_scale`
- **类型**：浮点数
- **默认值**：1.0
- **描述**：关节位置缩放因子
- **用法**：应与训练时的缩放匹配

#### `rbt_lin_vel_scale`
- **类型**：浮点数
- **默认值**：1.0
- **描述**：关节速度缩放因子

#### `rbt_ang_vel_scale`
- **类型**：浮点数
- **默认值**：1.0
- **描述**：基座角速度缩放因子

#### `clip_obs`
- **类型**：浮点数
- **默认值**：18.0
- **描述**：观测裁剪的最大绝对值
- **用法**：防止极端传感器值使策略不稳定

### 动作配置

#### `action_scale`
- **类型**：浮点数
- **默认值**：1.0
- **描述**：应用于所有策略输出的全局缩放因子
- **调优**：为安全起见，从较低值（0.5-0.7）开始，逐步增加

#### `clip_actions_lower`
- **类型**：12个浮点数的数组
- **默认值**：`[-1.00, -0.40, -0.60, -1.30, -0.75, -0.30, -1.00, -0.40, -0.60, -1.30, -0.75, -0.30]`
- **描述**：每个关节动作的下限（弧度）
- **安全性**：这些限制防止机器人超出安全关节范围

#### `clip_actions_upper`
- **类型**：12个浮点数的数组
- **默认值**：`[1.00, 0.40, 0.60, 1.30, 0.75, 0.30, 1.00, 0.40, 0.60, 1.30, 0.75, 0.30]`
- **描述**：每个关节动作的上限（弧度）

### 电机配置

#### `motor_direction`
- **类型**：12个整数的数组
- **默认值**：`[1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1]`
- **描述**：每个电机的方向乘数（1 或 -1）
- **用法**：校正电机安装方向差异

#### `map_index`
- **类型**：12个整数的数组
- **默认值**：`[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]`
- **描述**：将策略关节顺序映射到实际电机顺序
- **用法**：`actual_motor[i] = policy_joint[map_index[i]]`

#### `urdf_dof_pos_offset`
- **类型**：12个浮点数的数组
- **默认值**：`[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- **描述**：URDF 零位置与实际电机零位置之间的偏移
- **用法**：补偿校准差异

---

## 调优提示

### 提高行走稳定性
1. **从保守开始**：初始使用 `action_scale: 0.5`
2. **先在 STANDBY 模式下测试**（使用 `action_scale: 0.05`）
3. **逐步增加** action_scale 直到达到期望性能
4. **检查关节限制**：确保 `clip_actions_lower/upper` 匹配物理限制

### 速度控制
1. **调整指令缩放**：增加 `cmd_lin_vel_scale` / `cmd_ang_vel_scale` 以获得更灵敏的控制
2. **增量测试**：从慢速指令开始并逐渐增加

### 适配不同机器人
1. **更新 `motor_direction`**：匹配您机器人的电机安装
2. **更新 `map_index`**：匹配您机器人的关节顺序
3. **更新动作限制**：匹配您机器人的物理关节范围
4. **重新校准 `urdf_dof_pos_offset`**：如果您的机器人使用不同的零位置

---

## 配置示例

### 保守模式（安全测试）
```yaml
action_scale: 0.5
cmd_lin_vel_scale: 0.8
cmd_ang_vel_scale: 1.0
```

### 激进模式（快速响应）
```yaml
action_scale: 1.2
cmd_lin_vel_scale: 1.5
cmd_ang_vel_scale: 1.5
```

### 多策略设置
您可以维护多个配置文件：
```bash
config_conservative.yaml
config_normal.yaml
config_aggressive.yaml
```

通过修改启动文件加载不同配置：
```xml
<param name="config_file" value="$(find hightorque_rl_inference)/config_aggressive.yaml"/>
```

---

**高擎机电（HighTorque Robotics）**  
**最后更新**：2025年11月

