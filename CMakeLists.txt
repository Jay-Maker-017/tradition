cmake_minimum_required(VERSION 3.10)

# 生成 compile_commands.json（clangd / IntelliSense 依赖它获取头文件路径）
# 必须在 project() 之前设置，否则不会生成
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tradition)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# 灯条检测程序：main + 三个算法模块（检测 / 配对 / 可视化）
add_executable(main
    src/main.cpp
    src/lightbar.cpp
    src/armor.cpp
    src/visualize.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main ${OpenCV_LIBS})