cmake_minimum_required(VERSION 3.16)
project(CameraCalibration)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 创建可执行文件
add_executable(camera_calibration calibration_working.cpp)

# 链接OpenCV库
target_link_libraries(camera_calibration ${OpenCV_LIBS})

# 设置输出目录
set_target_properties(camera_calibration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Windows特定设置
if(WIN32)
    # 复制OpenCV DLL到输出目录
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OpenCV_SUFFIX "d")
    else()
        set(OpenCV_SUFFIX "")
    endif()
    
    # 设置调试工作目录
    set_target_properties(camera_calibration PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV包含目录: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV库: ${OpenCV_LIBS}")
