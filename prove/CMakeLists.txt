cmake_minimum_required(VERSION 3.5)

project(detect_test)
set(CMAKE_CXX_STANDARD 20)
include_directories(
    #OpenVINO推理引擎的头文件
    "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/inference_engine/include/"
    "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/inference_engine/include/ngraph/include/"
)

#查找必要的依赖包
set(OpenCV_DIR  "D:/Program Files/Intel/openvino_2021.4.752/opencv/cmake")
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/ngraph/lib/ngraph.dll")
set(ngraph_INCLUDE_DIRS "D:/Program Files/Intel/openvino_2021.4.752/deployment_tools/ngraph/include/")


# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/sjtu.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
#        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
        ${OpenCV_LIBS}
        detector
)