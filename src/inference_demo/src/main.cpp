#include "inference_demo/inference_demo.h"
#include <signal.h>

std::shared_ptr<inference_demo::InferenceDemo> g_demo;

void signalHandler(int sig)
{
    ROS_INFO("Received signal %d, shutting down...", sig);
    if (g_demo)
    {
        g_demo->stop();
    }
    ros::shutdown();
}

int main(int argc, char** argv)
{
    ROS_INFO("=== Main function started ===");
    ros::init(argc, argv, "inference_demo");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    ROS_INFO("ROS node initialized");

    signal(SIGINT, signalHandler);
    ROS_INFO("Signal handler registered");

    g_demo = std::make_shared<inference_demo::InferenceDemo>(nh);
    ROS_INFO("InferenceDemo object created");
    
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

