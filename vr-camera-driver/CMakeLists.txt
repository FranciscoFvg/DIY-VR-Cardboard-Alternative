cmake_minimum_required(VERSION 3.15)
project(CameraVR VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenVR SDK
set(OPENVR_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/openvr" CACHE PATH "Path to OpenVR SDK")
if(NOT EXISTS ${OPENVR_SDK_PATH})
    message(FATAL_ERROR "OpenVR SDK não encontrado. Baixe em: https://github.com/ValveSoftware/openvr")
endif()

include_directories(${OPENVR_SDK_PATH}/headers)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    link_directories(${OPENVR_SDK_PATH}/lib/win64)
else()
    link_directories(${OPENVR_SDK_PATH}/lib/win32)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/driver
)

# Vision Library (processamento de visão)
add_library(CameraVRVision STATIC
    src/PoseEstimator.cpp
    src/PoseFilter.cpp
    src/HandPoseReceiver.cpp
)

target_link_libraries(CameraVRVision
    Eigen3::Eigen
    ws2_32
)

# Driver OpenVR (DLL)
add_library(driver_cameravr SHARED
    driver/CameraVRDriver.cpp
    driver/VirtualController.cpp
)

target_link_libraries(driver_cameravr
    CameraVRVision
    openvr_api
)

set_target_properties(driver_cameravr PROPERTIES
    OUTPUT_NAME "driver_cameravr"
    PREFIX ""
)

# Aplicação principal
add_executable(CameraVRApp
    src/main.cpp
    driver/CameraVRDriver.cpp
    driver/VirtualController.cpp
)

target_link_libraries(CameraVRApp
    CameraVRVision
    openvr_api
)

# Instalar
install(TARGETS driver_cameravr
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
)

install(TARGETS CameraVRApp
    RUNTIME DESTINATION bin
)
