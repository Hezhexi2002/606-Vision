cmake_minimum_required(VERSION 3.0.2)
project(robot_camera)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV 4 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosconsole
  camera_info_manager
  image_transport
  # image_transport_plugins
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_camera
#  CATKIN_DEPENDS camera_info_manager image_transport image_transport_plugins sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  mvsdk/include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME} src/mv_camera_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME} PUBLIC mvsdk/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME} PUBLIC mvsdk/lib/arm64)
else()
  message(FATAL_ERROR "Unsupport host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
   MVSDK
   ${catkin_LIBRARIES}
 )

