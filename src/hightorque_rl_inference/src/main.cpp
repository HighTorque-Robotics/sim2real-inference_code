/**
 * @file main.cpp
 * @brief Entry point for HighTorque RL inference node
 *        高擎机电强化学习推理节点入口
 * 
 * This file contains the main function that initializes the ROS node and
 * creates the HighTorqueRLInference object. It handles signal interrupts (Ctrl+C)
 * for graceful shutdown.
 * 
 * 本文件包含初始化 ROS 节点并创建 HighTorqueRLInference 对象的主函数。
 * 它处理信号中断（Ctrl+C）以实现优雅关闭。
 * 
 * @author 高擎机电 (HighTorque Robotics)
 * @date 2025
 * @copyright Copyright (c) 2025 高擎机电 (HighTorque Robotics)
 */

#include "hightorque_rl_inference/hightorque_rl_inference.h"
#include <signal.h>

std::shared_ptr<hightorque_rl_inference::HighTorqueRLInference> g_demo;

/**
 * @brief Signal handler for graceful shutdown
 *        优雅关闭的信号处理器
 * 
 * @param sig Signal number / 信号编号
 */
void signalHandler(int sig)
{
    ROS_INFO("Received signal %d, shutting down...", sig);
    if (g_demo)
    {
        g_demo->stop();
    }
    ros::shutdown();
}

/**
 * @brief Main entry point
 *        主入口点
 * 
 * Initializes ROS node, creates HighTorqueRLInference instance, and starts the
 * inference loop.
 * 
 * 初始化 ROS 节点，创建 HighTorqueRLInference 实例，并启动推理循环。
 * 
 * @param argc Argument count / 参数数量
 * @param argv Argument values / 参数值
 * @return int Exit code / 退出代码
 */
int main(int argc, char** argv)
{
    ROS_INFO("=== Main function started ===");
    ros::init(argc, argv, "hightorque_rl_inference_node");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    ROS_INFO("ROS node initialized");

    signal(SIGINT, signalHandler);
    ROS_INFO("Signal handler registered");

    g_demo = std::make_shared<hightorque_rl_inference::HighTorqueRLInference>(nh);
    ROS_INFO("HighTorqueRLInference object created");
    
    if (!g_demo->init())
    {
        ROS_ERROR("Initialization failed!");
        return -1;
    }
    ROS_INFO("Initialization successful, starting run loop");
    
    g_demo->run();
    ROS_INFO("=== Main function completed ===");
    return 0;
}

