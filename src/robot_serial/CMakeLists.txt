cmake_minimum_required(VERSION 2.8.3)
project(robot_serial)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  serial
  std_msgs
  robot_msgs
)

# catkin_package(
#   CATKIN_DEPENDS
#     serial
#     std_msgs
#     robot_msgs
# )

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(robot_serial src/serial_node.cpp)

target_link_libraries(robot_serial
  ${catkin_LIBRARIES}
)

