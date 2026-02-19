cmake_minimum_required(VERSION 3.10)
project(HumanDetectionTest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Collect cpp files and target them for compilation, then add our headers
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Dependencies
find_package(OpenCV REQUIRED)
find_package(OnnxRuntime REQUIRED)

# Link to our libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
	${OpenCV_LIBS}
	OnnxRuntime::OnnxRuntime
)

# Create a directory for the model cache
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/model_cache")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/sam3-onnx" 
	DESTINATION "${CMAKE_BINARY_DIR}"
)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/language.token"
	DESTINATION "${CMAKE_BINARY_DIR}"
)
