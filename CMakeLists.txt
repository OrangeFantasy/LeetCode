cmake_minimum_required(VERSION 3.25)

project(YoloV5 LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/Source/*.h ${PROJECT_SOURCE_DIR}/Source/*.hpp)
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/Source/*.cpp)
file(GLOB_RECURSE CUDA_FILES ${PROJECT_SOURCE_DIR}/Source/*.cu)
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_FILES})

set(ThirdParty_DIR "E:/third_party")

set(CUDA_Toolkit_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
set(CUDNN_DIR "${ThirdParty_DIR}/cudnn-windows-x86_64-8.8.0.121_cuda12-archive")
set(TensorRT_DIR "${ThirdParty_DIR}/TensorRT-8.6.0.12")
set(OpenCV_DIR "${ThirdParty_DIR}/opencv-4.7.0-windows/build")

# find_package(OpenCV REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/Source
	${CUDA_Toolkit_DIR}/include
	${CUDNN_DIR}/include
	${TensorRT_DIR}/include
	${OpenCV_DIR}/include
)

link_directories(
	${CUDNN_DIR}/lib
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} ${CUDA_FILES})
#add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${CUDA_FILES})

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_DIR}/x64/vc16/lib/opencv_world470d.lib
	#${OpenCV_DIR}/x64/vc16/lib/opencv_world470.lib
)
target_link_libraries(${PROJECT_NAME} 
	${TensorRT_DIR}/lib/nvinfer.lib
	${TensorRT_DIR}/lib/nvinfer_dispatch.lib
	${TensorRT_DIR}/lib/nvinfer_lean.lib
	${TensorRT_DIR}/lib/nvinfer_plugin.lib
	${TensorRT_DIR}/lib/nvinfer_vc_plugin.lib
	${TensorRT_DIR}/lib/nvonnxparser.lib
	${TensorRT_DIR}/lib/nvparsers.lib
)
target_link_libraries(${PROJECT_NAME} 
	${CUDA_Toolkit_DIR}/lib/x64/cublas.lib
	${CUDA_Toolkit_DIR}/lib/x64/cublasLt.lib
	${CUDA_Toolkit_DIR}/lib/x64/cuda.lib
	${CUDA_Toolkit_DIR}/lib/x64/cudadevrt.lib
	${CUDA_Toolkit_DIR}/lib/x64/cudart.lib
	${CUDA_Toolkit_DIR}/lib/x64/cudart_static.lib
	${CUDA_Toolkit_DIR}/lib/x64/cufft.lib
	${CUDA_Toolkit_DIR}/lib/x64/cufftw.lib
	${CUDA_Toolkit_DIR}/lib/x64/curand.lib
	${CUDA_Toolkit_DIR}/lib/x64/cusolver.lib
	${CUDA_Toolkit_DIR}/lib/x64/cusparse.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppc.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppial.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppicc.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppidei.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppif.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppig.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppim.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppist.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppisu.lib
	${CUDA_Toolkit_DIR}/lib/x64/nppitc.lib
	${CUDA_Toolkit_DIR}/lib/x64/npps.lib
	${CUDA_Toolkit_DIR}/lib/x64/nvblas.lib
	${CUDA_Toolkit_DIR}/lib/x64/nvml.lib
	${CUDA_Toolkit_DIR}/lib/x64/nvrtc.lib
)

#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Build")
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Build")
SET(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Build")

