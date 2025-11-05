# Troubleshooting Guide / 故障排除指南

**Developed by 高擎机电 (HighTorque Robotics)**

[English](#english) | [中文](#中文)

---

## English

This guide helps you diagnose and fix common issues with the HighTorque RL Custom Inference Demo.

### 1. "Timeout waiting for robot data"

**Symptoms:**
```
[ WARN] Timeout waiting for robot data! stateReceived=0, imuReceived=1
```

**Causes:**
- Robot is not running or not in developer mode
- ROS topics are not being published
- Topic names don't match

**Solutions:**

1. **Check if robot is running:**
   ```bash
   rostopic list
   ```
   You should see:
   - `/sim2real_master_node/rbt_state`
   - `/sim2real_master_node/mtr_state`
   - `/imu/data`

2. **Verify topic data:**
   ```bash
   rostopic echo /sim2real_master_node/rbt_state
   ```
   Should show joint position and velocity data

3. **Check topic rate:**
   ```bash
   rostopic hz /sim2real_master_node/rbt_state
   ```
   Should be publishing at a consistent rate

4. **If topics are missing**, start the robot control system:
   - Ensure robot is powered on
   - Start developer mode on the robot controller
   - Verify network connection between robot and computer

---

### 2. "Model loading failed" / "invalid RKNN_MAGIC"

**Symptoms:**
```
E RKNN: invalid RKNN_MAGIC!
E RKNN: parseRKNN from buffer: Invalid RKNN format!
```

**Causes:**
- Model file is wrong format (e.g., `.pt` instead of `.rknn`)
- Model file is corrupted
- Model file path is incorrect

**Solutions:**

1. **Check model format:**
   ```bash
   ls -lh src/hightorque_rl_inference/policy/
   file src/hightorque_rl_inference/policy/your_model.rknn
   ```
   Must be a valid RKNN model file

2. **Verify model path in config:**
   ```bash
   cat src/hightorque_rl_inference/config_example.yaml | grep policy_name
   ```
   Should match an actual `.rknn` file in `policy/` directory

3. **Convert PyTorch model to RKNN:**
   - If you have a `.pt` or `.pth` file, you need to convert it
   - See [Development Guide](development.md#model-conversion) for conversion steps

4. **Re-download or re-convert model** if corrupted

---

### 3. "Joint state size mismatch"

**Symptoms:**
```
[ WARN] Joint state size mismatch: received 22, expected 12
```

**Causes:**
- Robot publishes more joints than the policy expects
- Configuration mismatch

**Solutions:**

1. **Check configuration:**
   ```yaml
   # config_example.yaml
   num_actions: 12  # Should match your policy
   ```

2. **The code automatically handles extra joints** by using only the first `num_actions` joints

3. **If you need different joint count:**
   - Update `num_actions` in config
   - Ensure your policy was trained with this joint count
   - Update `clip_actions_lower/upper` arrays

---

### 4. Robot behaves erratically or unstable

**Symptoms:**
- Robot shakes or oscillates
- Joints move to extreme positions
- Robot falls frequently

**Solutions:**

1. **Reduce action scale:**
   ```yaml
   # config_example.yaml
   action_scale: 0.5  # Start lower, increase gradually
   ```

2. **Check motor direction:**
   ```yaml
   motor_direction: [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1]
   ```
   - If a joint moves opposite to expected, flip its sign (1 → -1 or -1 → 1)

3. **Verify joint limits:**
   ```yaml
   clip_actions_lower: [-1.00, -0.40, ...]
   clip_actions_upper: [1.00, 0.40, ...]
   ```
   - Ensure limits match physical robot constraints
   - Make limits more conservative if needed

4. **Check map_index:**
   ```yaml
   map_index: [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
   ```
   - Incorrect mapping causes wrong joints to receive commands
   - Verify this matches your robot's joint ordering

5. **Test in STANDBY mode first:**
   - Press `LT + RT + START` to enter STANDBY
   - Uses minimal action scale (0.05) for safe testing
   - Observe which joints move incorrectly

---

### 5. "Could not load YAML config"

**Symptoms:**
```
[ERROR] Could not load YAML config: ...
[ WARN] Using default parameters
```

**Causes:**
- Config file doesn't exist or path is wrong
- YAML syntax error
- File permissions issue

**Solutions:**

1. **Verify file exists:**
   ```bash
   ls -l src/hightorque_rl_inference/config_example.yaml
   ```

2. **Check YAML syntax:**
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('src/hightorque_rl_inference/config_example.yaml'))"
   ```
   Should not print any errors

3. **Fix common YAML mistakes:**
   - Ensure proper indentation (use spaces, not tabs)
   - Check array format: `[1, 2, 3]` or list format with dashes
   - Ensure colons have space after them: `key: value`

4. **Check file permissions:**
   ```bash
   chmod 644 src/hightorque_rl_inference/config_example.yaml
   ```

---

### 6. Node crashes or segmentation fault

**Symptoms:**
- Node exits with `Segmentation fault (core dumped)`
- Sudden crash during inference

**Possible Causes:**
- RKNN library issue
- Memory corruption
- Model/code mismatch

**Solutions:**

1. **Check RKNN library:**
   ```bash
   ldd devel/lib/hightorque_rl_inference/hightorque_rl_inference_node | grep rknn
   ```
   Should show `librknnrt.so` found

2. **Verify model dimensions:**
   - Input: Should be `num_single_obs * frame_stack`
   - Output: Should be `num_actions`
   - Mismatch causes buffer overflows

3. **Enable debug output:**
   ```bash
   roslaunch hightorque_rl_inference hightorque_rl_inference.launch --screen
   ```
   Look for last message before crash

4. **Run with debugger:**
   ```bash
   rosrun hightorque_rl_inference hightorque_rl_inference_node __log:=screen
   ```

---

### 7. Robot doesn't respond to /cmd_vel

**Symptoms:**
- Publishing to `/cmd_vel` has no effect
- Robot stays still even in RUNNING mode

**Solutions:**

1. **Verify you're in RUNNING mode:**
   - State machine must be: `NOT_READY → STANDBY → RUNNING`
   - Press `LT + RT + LB` to transition from STANDBY to RUNNING

2. **Check topic connection:**
   ```bash
   rostopic info /cmd_vel
   ```
   Should show `hightorque_rl_inference_node` as a subscriber

3. **Test with simple command:**
   ```bash
   rostopic pub /cmd_vel geometry_msgs/Twist \
     "linear: {x: 0.2, y: 0.0, z: 0.0}
      angular: {x: 0.0, y: 0.0, z: 0.0}"
   ```

4. **Check velocity scaling:**
   ```yaml
   cmd_lin_vel_scale: 1.0   # Increase for more sensitivity
   cmd_ang_vel_scale: 1.25
   ```

---

### 8. Joystick state transitions not working

**Symptoms:**
- Pressing button combinations has no effect
- Can't transition between states

**Solutions:**

1. **Verify joystick is connected:**
   ```bash
   rostopic echo /joy
   ```
   Press buttons and check if values change

2. **Check button mapping:**
   - `LT + RT + START`: NOT_READY → STANDBY
   - `LT + RT + LB`: STANDBY ↔ RUNNING
   - Button indices depend on your joystick model

3. **Debug button indices:**
   ```bash
   rostopic echo /joy
   ```
   Note which indices change when you press LT, RT, START, LB

4. **If using different joystick**, modify `run()` function button indices in code

---

## Debugging Tips

### Enable verbose logging

In `src/hightorque_rl_inference/src/hightorque_rl_inference.cpp`, look for `ROS_DEBUG` statements.

Change to `ROS_INFO` for more output:
```cpp
ROS_DEBUG(...) → ROS_INFO(...)
```

### Check all topic rates

```bash
rostopic hz /sim2real_master_node/rbt_state
rostopic hz /sim2real_master_node/mtr_state
rostopic hz /imu/data
rostopic hz /pi_plus_all
```

All should be publishing at expected rates.

### Monitor system resources

```bash
htop
```

Check CPU and memory usage. RKNN inference should use minimal CPU.

### Test with minimal config

Create a minimal `config_test.yaml` with conservative settings:
```yaml
num_actions: 12
num_single_obs: 36
frame_stack: 1
action_scale: 0.3  # Very conservative
clip_obs: 18.0
policy_name: "your_model.rknn"
dt: 0.001
decimation: 10
# ... rest with defaults
```

---

## Getting Help

If issues persist:

1. **Check GitHub Issues**: [Repository Issues](https://github.com/HighTorque-Robotics/hightorque_rl_custom/issues)
2. **Collect information**:
   - ROS version: `rosversion -d`
   - Package version: `rospack find hightorque_rl_inference`
   - Config file contents
   - Full error log
3. **Contact support**: support@hightorque-robotics.com

---

## 中文

本指南帮助您诊断和修复 HighTorque RL 自定义推理演示的常见问题。

### 1. "等待机器人数据超时"

**症状：**
```
[ WARN] Timeout waiting for robot data! stateReceived=0, imuReceived=1
```

**原因：**
- 机器人未运行或未处于开发者模式
- ROS 话题未发布
- 话题名称不匹配

**解决方案：**

1. **检查机器人是否运行：**
   ```bash
   rostopic list
   ```
   您应该看到：
   - `/sim2real_master_node/rbt_state`
   - `/sim2real_master_node/mtr_state`
   - `/imu/data`

2. **验证话题数据：**
   ```bash
   rostopic echo /sim2real_master_node/rbt_state
   ```
   应显示关节位置和速度数据

3. **检查话题频率：**
   ```bash
   rostopic hz /sim2real_master_node/rbt_state
   ```
   应以一致的频率发布

4. **如果话题缺失**，启动机器人控制系统：
   - 确保机器人已通电
   - 在机器人控制器上启动开发者模式
   - 验证机器人与计算机之间的网络连接

---

### 2. "模型加载失败" / "无效的 RKNN_MAGIC"

**症状：**
```
E RKNN: invalid RKNN_MAGIC!
E RKNN: parseRKNN from buffer: Invalid RKNN format!
```

**原因：**
- 模型文件格式错误（例如 `.pt` 而不是 `.rknn`）
- 模型文件损坏
- 模型文件路径不正确

**解决方案：**

1. **检查模型格式：**
   ```bash
   ls -lh src/hightorque_rl_inference/policy/
   file src/hightorque_rl_inference/policy/your_model.rknn
   ```
   必须是有效的 RKNN 模型文件

2. **验证配置中的模型路径：**
   ```bash
   cat src/hightorque_rl_inference/config_example.yaml | grep policy_name
   ```
   应匹配 `policy/` 目录中的实际 `.rknn` 文件

3. **将 PyTorch 模型转换为 RKNN：**
   - 如果您有 `.pt` 或 `.pth` 文件，需要进行转换
   - 参见[开发指南](development.md#模型转换)了解转换步骤

4. **如果损坏，重新下载或重新转换模型**

---

### 3. "关节状态大小不匹配"

**症状：**
```
[ WARN] Joint state size mismatch: received 22, expected 12
```

**原因：**
- 机器人发布的关节数多于策略期望的数量
- 配置不匹配

**解决方案：**

1. **检查配置：**
   ```yaml
   # config_example.yaml
   num_actions: 12  # 应匹配您的策略
   ```

2. **代码会自动处理额外的关节**，仅使用前 `num_actions` 个关节

3. **如果需要不同的关节数：**
   - 更新配置中的 `num_actions`
   - 确保策略是用此关节数训练的
   - 更新 `clip_actions_lower/upper` 数组

---

### 4. 机器人行为异常或不稳定

**症状：**
- 机器人抖动或振荡
- 关节移动到极端位置
- 机器人频繁跌倒

**解决方案：**

1. **降低动作缩放：**
   ```yaml
   # config_example.yaml
   action_scale: 0.5  # 从较低值开始，逐渐增加
   ```

2. **检查电机方向：**
   ```yaml
   motor_direction: [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1]
   ```
   - 如果关节移动方向与预期相反，翻转其符号（1 → -1 或 -1 → 1）

3. **验证关节限制：**
   ```yaml
   clip_actions_lower: [-1.00, -0.40, ...]
   clip_actions_upper: [1.00, 0.40, ...]
   ```
   - 确保限制匹配物理机器人约束
   - 如需要，使限制更保守

4. **检查 map_index：**
   ```yaml
   map_index: [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
   ```
   - 映射不正确会导致错误的关节接收指令
   - 验证这与您机器人的关节顺序匹配

5. **先在 STANDBY 模式下测试：**
   - 按 `LT + RT + START` 进入 STANDBY
   - 使用最小动作缩放（0.05）进行安全测试
   - 观察哪些关节移动不正确

---

### 5. "无法加载 YAML 配置"

**症状：**
```
[ERROR] Could not load YAML config: ...
[ WARN] Using default parameters
```

**原因：**
- 配置文件不存在或路径错误
- YAML 语法错误
- 文件权限问题

**解决方案：**

1. **验证文件存在：**
   ```bash
   ls -l src/hightorque_rl_inference/config_example.yaml
   ```

2. **检查 YAML 语法：**
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('src/hightorque_rl_inference/config_example.yaml'))"
   ```
   不应打印任何错误

3. **修复常见 YAML 错误：**
   - 确保正确缩进（使用空格，不使用制表符）
   - 检查数组格式：`[1, 2, 3]` 或带破折号的列表格式
   - 确保冒号后有空格：`key: value`

4. **检查文件权限：**
   ```bash
   chmod 644 src/hightorque_rl_inference/config_example.yaml
   ```

---

### 6. 节点崩溃或段错误

**症状：**
- 节点以 `Segmentation fault (core dumped)` 退出
- 推理期间突然崩溃

**可能原因：**
- RKNN 库问题
- 内存损坏
- 模型/代码不匹配

**解决方案：**

1. **检查 RKNN 库：**
   ```bash
   ldd devel/lib/hightorque_rl_inference/hightorque_rl_inference_node | grep rknn
   ```
   应显示找到 `librknnrt.so`

2. **验证模型维度：**
   - 输入：应为 `num_single_obs * frame_stack`
   - 输出：应为 `num_actions`
   - 不匹配会导致缓冲区溢出

3. **启用调试输出：**
   ```bash
   roslaunch hightorque_rl_inference hightorque_rl_inference.launch --screen
   ```
   查找崩溃前的最后一条消息

4. **使用调试器运行：**
   ```bash
   rosrun hightorque_rl_inference hightorque_rl_inference_node __log:=screen
   ```

---

### 7. 机器人不响应 /cmd_vel

**症状：**
- 向 `/cmd_vel` 发布没有效果
- 即使在 RUNNING 模式下机器人也保持静止

**解决方案：**

1. **验证处于 RUNNING 模式：**
   - 状态机必须是：`NOT_READY → STANDBY → RUNNING`
   - 按 `LT + RT + LB` 从 STANDBY 转换到 RUNNING

2. **检查话题连接：**
   ```bash
   rostopic info /cmd_vel
   ```
   应显示 `hightorque_rl_inference_node` 为订阅者

3. **使用简单命令测试：**
   ```bash
   rostopic pub /cmd_vel geometry_msgs/Twist \
     "linear: {x: 0.2, y: 0.0, z: 0.0}
      angular: {x: 0.0, y: 0.0, z: 0.0}"
   ```

4. **检查速度缩放：**
   ```yaml
   cmd_lin_vel_scale: 1.0   # 增加以提高灵敏度
   cmd_ang_vel_scale: 1.25
   ```

---

### 8. 手柄状态转换不工作

**症状：**
- 按下按钮组合没有效果
- 无法在状态之间转换

**解决方案：**

1. **验证手柄已连接：**
   ```bash
   rostopic echo /joy
   ```
   按下按钮并检查值是否变化

2. **检查按钮映射：**
   - `LT + RT + START`：NOT_READY → STANDBY
   - `LT + RT + LB`：STANDBY ↔ RUNNING
   - 按钮索引取决于您的手柄型号

3. **调试按钮索引：**
   ```bash
   rostopic echo /joy
   ```
   注意按下 LT、RT、START、LB 时哪些索引变化

4. **如果使用不同的手柄**，修改代码中 `run()` 函数的按钮索引

---

## 调试提示

### 启用详细日志

在 `src/hightorque_rl_inference/src/hightorque_rl_inference.cpp` 中，查找 `ROS_DEBUG` 语句。

更改为 `ROS_INFO` 以获得更多输出：
```cpp
ROS_DEBUG(...) → ROS_INFO(...)
```

### 检查所有话题频率

```bash
rostopic hz /sim2real_master_node/rbt_state
rostopic hz /sim2real_master_node/mtr_state
rostopic hz /imu/data
rostopic hz /pi_plus_all
```

所有话题应以预期频率发布。

### 监控系统资源

```bash
htop
```

检查 CPU 和内存使用情况。RKNN 推理应使用最少的 CPU。

### 使用最小配置测试

创建一个保守设置的最小 `config_test.yaml`：
```yaml
num_actions: 12
num_single_obs: 36
frame_stack: 1
action_scale: 0.3  # 非常保守
clip_obs: 18.0
policy_name: "your_model.rknn"
dt: 0.001
decimation: 10
# ... 其余使用默认值
```

---

## 获取帮助

如果问题仍然存在：

1. **检查 GitHub Issues**：[仓库问题](https://github.com/HighTorque-Robotics/hightorque_rl_custom/issues)
2. **收集信息**：
   - ROS 版本：`rosversion -d`
   - 功能包版本：`rospack find hightorque_rl_inference`
   - 配置文件内容
   - 完整错误日志
3. **联系支持**：support@hightorque-robotics.com

---

**高擎机电（HighTorque Robotics）**  
**最后更新**：2025年11月

