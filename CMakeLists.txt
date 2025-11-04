cmake_minimum_required(VERSION 3.15)
project(EndoRobo_EnvAwareNav VERSION 1.0.0 LANGUAGES CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /WX-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 查找Python (用于嵌入Python解释器)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# yaml-cpp for configuration
find_package(yaml-cpp REQUIRED)

# PCL (Point Cloud Library) for 3D reconstruction
# Made optional - can use system PCL 1.9+ or vcpkg version
# Temporarily disabled - install via vcpkg if needed
# find_package(PCL 1.9 QUIET COMPONENTS common io filters surface visualization)
set(PCL_FOUND FALSE)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# Add NumPy include directory
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NUMPY_INCLUDE_DIR)
        include_directories(${NUMPY_INCLUDE_DIR})
        message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")
    endif()
endif()

# PCL is optional
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    message(STATUS "PCL ${PCL_VERSION} found - 3D reconstruction enabled")
else()
    message(WARNING "PCL not found - 3D reconstruction features disabled")
endif()

# 源文件
set(CORE_SOURCES
    src/core/config_manager.cpp
    src/core/logger.cpp
)

set(CAMERA_SOURCES
    src/camera/camera_capture.cpp
    src/camera/image_processor.cpp
)

set(PYTHON_INTERFACE_SOURCES
    src/python_interface/python_wrapper.cpp
    src/python_interface/pose_estimator.cpp
    src/python_interface/depth_estimator_dav2.cpp
)

set(RECONSTRUCTION_SOURCES
    # PCL-dependent files temporarily disabled
    # src/reconstruction/point_cloud_builder.cpp
    # src/reconstruction/intestinal_reconstructor.cpp
    # src/reconstruction/redundancy_remover.cpp
)

set(UTILS_SOURCES
    src/utils/timer.cpp
    src/utils/math_utils.cpp
)

# 创建库
add_library(endorobo_core STATIC
    ${CORE_SOURCES}
    ${CAMERA_SOURCES}
    ${PYTHON_INTERFACE_SOURCES}
    ${RECONSTRUCTION_SOURCES}
    ${UTILS_SOURCES}
)

# 链接库
target_link_libraries(endorobo_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${Python3_LIBRARIES}
    yaml-cpp
)

# Link PCL if found
if(PCL_FOUND)
    target_link_libraries(endorobo_core ${PCL_LIBRARIES})
endif()

# 主程序
add_executable(endorobo_main
    src/main.cpp
)

target_link_libraries(endorobo_main
    endorobo_core
)

# 安装规则
install(TARGETS endorobo_main endorobo_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY config/
    DESTINATION config
    FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml"
)

# 打印配置信息
message(STATUS "====================================")
message(STATUS "EndoRobo Environment-Aware Navigation")
message(STATUS "====================================")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "====================================")

