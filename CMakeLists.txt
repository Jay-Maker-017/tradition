cmake_minimum_required(VERSION 3.10)

# 生成 compile_commands.json（clangd / IntelliSense 依赖它获取头文件路径）
# 必须在 project() 之前设置，否则不会生成
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tradition)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# ==================== OpenVINO 查找（NNdeployment 依赖） ====================
# 优先使用系统安装的 OpenVINO（apt 安装后 CMake 可自动找到）；
# 找不到时退回使用 pip 安装的 openvino 包自带的 C++ 开发文件（无需 root 权限）。
if(NOT OpenVINO_DIR)
    execute_process(
        COMMAND python3 -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))"
        OUTPUT_VARIABLE TRADITION_PIP_OPENVINO_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(TRADITION_PIP_OPENVINO_DIR AND EXISTS "${TRADITION_PIP_OPENVINO_DIR}/OpenVINOConfig.cmake")
        set(OpenVINO_DIR "${TRADITION_PIP_OPENVINO_DIR}" CACHE PATH "OpenVINO CMake 配置目录（pip 安装的 openvino 包）")
        message(STATUS "OpenVINO: 使用 pip 安装包 ${TRADITION_PIP_OPENVINO_DIR}")
    endif()
endif()


# ==================== 神经网络部署库（迁移自 26_Detection_Models-main） ====================
# 数字神经网络识别装甲板类型所用的统一部署库（OpenVINO 后端）
add_subdirectory(NNdeployment)


# 灯条检测程序：main 应用编排 + 算法检测、数字识别、报告与可视化模块
add_executable(main
    src/main.cpp
    src/lightbar.cpp
    src/armor.cpp
    src/digit_recognizer.cpp
    src/visualize.cpp
    src/report.cpp
    src/pipeline.cpp
    # PnP 位姿解算（仅照片模式）
    src/pnp_solver.cpp
    src/pnp_verifier.cpp)
target_include_directories(main PRIVATE include)
# NNdeployment 公开接口需要 C++20（GCC 11 及以上支持）
target_compile_features(main PRIVATE cxx_std_20)
target_link_libraries(main ${OpenCV_LIBS} NNdeployment_lib)