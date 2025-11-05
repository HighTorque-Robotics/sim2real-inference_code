# ================================================================================
# CMakeLists.txt for hightorque_rl_inference package
# hightorque_rl_inference 功能包的 CMake 构建配置文件
#
# This file defines the build configuration for the RL inference demo package.
# It sets up dependencies, include directories, and build targets for both the
# library and executable node.
# 
# 本文件定义了强化学习推理演示功能包的构建配置。
# 它设置了依赖项、包含目录，以及库和可执行节点的构建目标。
#
# Developed by: 高擎机电 (HighTorque Robotics)
# ================================================================================

cmake_minimum_required(VERSION 3.0.2)
project(hightorque_rl_inference)

# ================================================================================
# Compiler Options / 编译器选项
# ================================================================================
add_compile_options(-std=c++17)

# ================================================================================
# Find Catkin Dependencies / 查找 Catkin 依赖项
# ================================================================================
# Required ROS packages for building this package
# 构建此功能包所需的 ROS 包
find_package(catkin REQUIRED COMPONENTS
  roscpp          # ROS C++ client library / ROS C++ 客户端库
  sensor_msgs     # Sensor message types (JointState, Imu, Joy) / 传感器消息类型
  geometry_msgs   # Geometry message types (Twist) / 几何消息类型
  roslib          # ROS library utilities (package path) / ROS 库工具
)

# ================================================================================
# Find System Dependencies / 查找系统依赖项
# ================================================================================
# Eigen3: Linear algebra library for vector/matrix operations
# Eigen3: 用于向量/矩阵运算的线性代数库
find_package(Eigen3 REQUIRED)

# yaml-cpp: YAML parsing library for configuration files
# yaml-cpp: 用于配置文件的 YAML 解析库
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# ================================================================================
# Catkin Package Configuration / Catkin 功能包配置
# ================================================================================
# Declare what this package provides to other packages
# 声明此功能包向其他功能包提供的内容
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs
)

# ================================================================================
# Include Directories / 包含目录
# ================================================================================
# Specify where the compiler should look for header files
# 指定编译器应在哪里查找头文件
include_directories(
  include                      # Package headers / 功能包头文件
  ${catkin_INCLUDE_DIRS}       # Catkin package headers / Catkin 包头文件
  ${EIGEN3_INCLUDE_DIRS}       # Eigen3 headers / Eigen3 头文件
  ${YAMLCPP_INCLUDE_DIRS}      # yaml-cpp headers / yaml-cpp 头文件
)

# ================================================================================
# Platform-Specific Configuration / 平台特定配置
# ================================================================================
# Force ARM (RKNN) only build - this package is designed for ARM controllers
# 强制仅构建 ARM (RKNN) 版本 - 此功能包专为 ARM 控制器设计
add_definitions(-DPLATFORM_ARM)

# RKNN inference engine headers and libraries
# RKNN 推理引擎头文件和库
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/rknn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(INFERENCE_LIBS rknnrt)

# ================================================================================
# Build Targets / 构建目标
# ================================================================================

# --- Library Target / 库目标 ---
# Build the main inference library
# 构建主推理库
add_library(${PROJECT_NAME}
  src/hightorque_rl_inference.cpp
)

# Link the library with required dependencies
# 将库与所需依赖项链接
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}      # ROS libraries / ROS 库
  ${INFERENCE_LIBS}        # RKNN runtime / RKNN 运行时
  ${YAMLCPP_LIBRARIES}     # YAML parser / YAML 解析器
)

# --- Executable Target / 可执行目标 ---
# Build the ROS node executable
# 构建 ROS 节点可执行文件
add_executable(${PROJECT_NAME}_node
  src/main.cpp
)

# Link the executable with the library and ROS
# 将可执行文件与库和 ROS 链接
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# ================================================================================
# Install Rules / 安装规则
# ================================================================================
# Define how files should be installed when 'catkin_make install' is run
# 定义运行 'catkin_make install' 时文件应如何安装

# Install library and executable
# 安装库和可执行文件
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install header files
# 安装头文件
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install launch files
# 安装启动文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
