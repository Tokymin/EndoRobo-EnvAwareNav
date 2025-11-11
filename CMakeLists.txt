cmake_minimum_required(VERSION 3.15)
project(EndoRobo_EnvAwareNav VERSION 1.0.0 LANGUAGES CXX)

# vcpkg toolchain (if vcpkg is in the project directory)
# This allows CMake to find packages installed via vcpkg
if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /WX- /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# OpenCV configuration
# Try to find OpenCV from common installation paths
# This must be set BEFORE find_package(OpenCV)
set(OpenCV_DIR "" CACHE PATH "OpenCV directory")

# Check environment variable first
if(DEFINED ENV{OpenCV_DIR} AND EXISTS "$ENV{OpenCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "OpenCV directory" FORCE)
    message(STATUS "Using OpenCV_DIR from environment: $ENV{OpenCV_DIR}")
else()
    # Try common installation paths
    set(OPENCV_POSSIBLE_PATHS
        "D:/Program Files/opencv-4.3.0/opencv_cmake/install/x64/vc16/lib"
        "D:/Program Files/opencv-4.3.0/opencv_cmake/install/x64/vc16"
        "D:/Program Files/opencv-4.3.0/opencv_cmake/install/x64/vc16/lib/cmake"
        "C:/Program Files/opencv-4.3.0/opencv_cmake/install/x64/vc16/lib"
        "C:/opencv/build"
    )
    foreach(OPENCV_PATH ${OPENCV_POSSIBLE_PATHS})
        if(EXISTS "${OPENCV_PATH}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${OPENCV_PATH}" CACHE PATH "OpenCV directory" FORCE)
            message(STATUS "Found OpenCV at: ${OPENCV_PATH}")
            break()
        endif()
    endforeach()
endif()

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 查找Python (用于嵌入Python解释器)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# yaml-cpp for configuration
find_package(yaml-cpp REQUIRED)

# PCL (Point Cloud Library) for 3D reconstruction
# Installation options:
# 1. AllInOne installer: https://github.com/PointCloudLibrary/pcl/releases
# 2. vcpkg: vcpkg install pcl:x64-windows
# If PCL_ROOT environment variable is set, CMake will find it automatically

# Set VTK_DIR if PCL_ROOT is defined (for AllInOne installer)
# This must be set BEFORE find_package(PCL) to avoid VTK path conflicts
if(DEFINED ENV{PCL_ROOT})
    # Try to find VTK in PCL's 3rdParty directory
    set(VTK_POSSIBLE_DIRS
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-9.4"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-9.3"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-9.2"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-9.1"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-9.0"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-8.2"
        "$ENV{PCL_ROOT}/3rdParty/VTK/lib/vtk-8.1"
    )
    foreach(VTK_DIR_PATH ${VTK_POSSIBLE_DIRS})
        if(EXISTS "${VTK_DIR_PATH}/VTKConfig.cmake")
            set(VTK_DIR "${VTK_DIR_PATH}" CACHE PATH "VTK directory" FORCE)
            message(STATUS "Found VTK at: ${VTK_DIR_PATH}")
            break()
        endif()
    endforeach()
endif()

# Also use environment variable if set
if(DEFINED ENV{VTK_DIR} AND EXISTS "$ENV{VTK_DIR}/VTKConfig.cmake")
    set(VTK_DIR "$ENV{VTK_DIR}" CACHE PATH "VTK directory" FORCE)
    message(STATUS "Using VTK_DIR from environment: $ENV{VTK_DIR}")
endif()

# Try to find PCL from environment variable first
if(DEFINED ENV{PCL_ROOT} AND NOT PCL_FOUND)
    # AllInOne安装包的CMake配置可能在cmake或lib/cmake目录
    if(EXISTS "$ENV{PCL_ROOT}/cmake/PCLConfig.cmake")
        set(PCL_DIR "$ENV{PCL_ROOT}/cmake")
    elseif(EXISTS "$ENV{PCL_ROOT}/lib/cmake/pcl-1.15")
        set(PCL_DIR "$ENV{PCL_ROOT}/lib/cmake/pcl-1.15")
    endif()
    message(STATUS "Using PCL_ROOT: $ENV{PCL_ROOT}")
endif()

# Try common installation paths if not found
if(NOT PCL_FOUND)
    set(PCL_POSSIBLE_PATHS
        "C:/Program Files/PCL 1.15.0"
        "D:/Program Files/PCL 1.15.0"
        "C:/PCL 1.15.0"
        "D:/PCL 1.15.0"
        "C:/PCL"
        "D:/PCL"
    )
    foreach(PCL_PATH ${PCL_POSSIBLE_PATHS})
        if(EXISTS "${PCL_PATH}/cmake/PCLConfig.cmake")
            set(PCL_DIR "${PCL_PATH}/cmake")
            message(STATUS "Found PCL at: ${PCL_PATH}")
            break()
        elseif(EXISTS "${PCL_PATH}/lib/cmake/pcl-1.15")
            set(PCL_DIR "${PCL_PATH}/lib/cmake/pcl-1.15")
            message(STATUS "Found PCL at: ${PCL_PATH}")
            break()
        endif()
    endforeach()
endif()

find_package(PCL 1.9 QUIET COMPONENTS common io filters visualization surface features)
if(NOT PCL_FOUND)
    message(WARNING "PCL not found - 3D reconstruction features will be disabled")
    message(WARNING "Installation options:")
    message(WARNING "  1. Download AllInOne installer from: https://github.com/PointCloudLibrary/pcl/releases")
    message(WARNING "  2. Use vcpkg: vcpkg install pcl:x64-windows")
    set(PCL_FOUND FALSE)
else()
    message(STATUS "PCL ${PCL_VERSION} found - 3D reconstruction enabled")
endif()

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
    # PCL-dependent files - enabled when PCL is found
)
if(PCL_FOUND)
    list(APPEND RECONSTRUCTION_SOURCES
        src/reconstruction/point_cloud_builder.cpp
        src/reconstruction/intestinal_reconstructor.cpp
        src/reconstruction/redundancy_remover.cpp
    )
endif()

set(NAVIGATION_SOURCES
    src/navigation/visual_odometry.cpp
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
    ${NAVIGATION_SOURCES}
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
    # AllInOne安装包使用传统的PCL变量，而不是现代CMake目标
    # 检查是否有现代CMake目标可用
    if(TARGET PCL::common)
        # Use modern CMake target-based linking if available
        target_link_libraries(endorobo_core 
            PCL::common
            PCL::io
            PCL::filters
            PCL::visualization
        )
    else()
        # Fall back to traditional PCL variables (for AllInOne installer)
        target_link_libraries(endorobo_core 
            ${PCL_LIBRARIES}
        )
        # Also link required PCL components explicitly
        if(PCL_COMMON_LIBRARY)
            target_link_libraries(endorobo_core ${PCL_COMMON_LIBRARY})
        endif()
        if(PCL_IO_LIBRARY)
            target_link_libraries(endorobo_core ${PCL_IO_LIBRARY})
        endif()
        if(PCL_FILTERS_LIBRARY)
            target_link_libraries(endorobo_core ${PCL_FILTERS_LIBRARY})
        endif()
        if(PCL_VISUALIZATION_LIBRARY)
            target_link_libraries(endorobo_core ${PCL_VISUALIZATION_LIBRARY})
        endif()
    endif()
endif()

# 主程序
add_executable(endorobo_main
    src/main.cpp
)

target_link_libraries(endorobo_main
    endorobo_core
)

# PCL测试程序（可选）
if(PCL_FOUND)
    # Simple PCL test
    add_executable(test_pcl_simple
        test_pcl_simple.cpp
    )
    
    target_include_directories(test_pcl_simple PRIVATE
        ${PCL_INCLUDE_DIRS}
    )
    
    target_link_libraries(test_pcl_simple
        ${PCL_LIBRARIES}
    )
    
    # Also link required PCL components explicitly
    if(PCL_COMMON_LIBRARY)
        target_link_libraries(test_pcl_simple ${PCL_COMMON_LIBRARY})
    endif()
    
    # Full PCL test with IO
    add_executable(test_pcl
        test_pcl.cpp
    )
    
    target_include_directories(test_pcl PRIVATE
        ${PCL_INCLUDE_DIRS}
    )
    
    target_link_libraries(test_pcl
        ${PCL_LIBRARIES}
    )
    
    if(PCL_COMMON_LIBRARY)
        target_link_libraries(test_pcl ${PCL_COMMON_LIBRARY})
    endif()
    if(PCL_IO_LIBRARY)
        target_link_libraries(test_pcl ${PCL_IO_LIBRARY})
    endif()
    
    # Reconstruction test (no IO dependencies, avoids OpenNI2)
    add_executable(test_pcl_reconstruction
        test_pcl_reconstruction.cpp
    )
    
    target_include_directories(test_pcl_reconstruction PRIVATE
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
    )
    
    target_link_libraries(test_pcl_reconstruction
        ${PCL_LIBRARIES}
        Eigen3::Eigen
    )
    
    if(PCL_COMMON_LIBRARY)
        target_link_libraries(test_pcl_reconstruction ${PCL_COMMON_LIBRARY})
    endif()
    if(PCL_FILTERS_LIBRARY)
        target_link_libraries(test_pcl_reconstruction ${PCL_FILTERS_LIBRARY})
    endif()
    
    message(STATUS "PCL test programs 'test_pcl', 'test_pcl_simple', and 'test_pcl_reconstruction' added")
endif()

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

