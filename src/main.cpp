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
    ros::init(argc, argv, "inference_demo");
    auto nh = std::make_shared<ros::NodeHandle>("~");

    signal(SIGINT, signalHandler);

    g_demo = std::make_shared<inference_demo::InferenceDemo>(nh);
    if (!g_demo->init())
    {
        ROS_ERROR("Initialization failed!");
        return -1;
    }
    g_demo->run();
    return 0;
}

