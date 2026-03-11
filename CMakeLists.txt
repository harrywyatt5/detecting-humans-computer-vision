cmake_minimum_required(VERSION 3.24)
project(real_time_humans LANGUAGES CXX CUDA)
find_package(ament_cmake REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-reorder)
endif()


# Collect cpp files and target them for compilation, then add our headers
set(SOURCES
	src/main.cpp
	src/LanguageToken.cpp
	src/NormaliseImageKernel.cu
	src/PersistentImageInput.cpp
	src/PersistentSam3Model.cpp
	src/Sam3Context.cpp
	src/Sam3ContextBuilder.cpp
	src/TextEncoderSession.cpp
	src/TextEncoderSessionFactory.cpp
	src/PersistentImageInputFactory.cpp
	src/VisionEncoderSession.cpp
	src/VisionEncoderSessionFactory.cpp
	src/MaskDecoderSession.cpp
	src/MaskDecoderSessionFactory.cpp
	src/TextEncoderInitialiser.cpp
	src/VisionEncoderInitialiser.cpp
	src/MaskDecoderInitialiser.cpp
	src/CreateImageProcessor.cpp
	src/CreateImageKernel.cu
	src/Binding.cpp
	src/BindingBlueprint.cpp
	src/HumanDetectionNode.cpp
	src/LoggingLevel.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Dependencies
find_package(OpenCV 4 REQUIRED HINTS "/opt/cudaopencv")
find_package(CUDAToolkit REQUIRED)
find_package(OnnxRuntime REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Link standard CMAKE libs
target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	OnnxRuntime::OnnxRuntime
	CUDA::cudart
)

# Link special ROS2 libs
ament_target_dependencies(${PROJECT_NAME}
	rclcpp
	std_msgs
	sensor_msgs
	ament_index_cpp
)

# Create a directory for the model cache
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sam3-onnx"
	DESTINATION share/${PROJECT_NAME}
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/language.token"
	DESTINATION share/${PROJECT_NAME}
)
ament_package()
