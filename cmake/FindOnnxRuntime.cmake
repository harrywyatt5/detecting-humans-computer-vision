# Is a helper library for finding libraries on the system
include(FindPackageHandleStandardArgs)

file(GLOB ONNXRUNTIME_WC_INCLUDES 
	"/opt/onnx*/include" 
	"/usr/local/onnx*/include"
	"$ENV{HOME}/onnx*/include"
)
file(GLOB ONNXRUNTIME_WC_LIBS
	"/opt/onnx*/lib"
	"/usr/local/onnx*/lib"
	"$ENV{HOME}/onnx*/lib"
)

find_path(OnnxRuntime_INCLUDE_DIR
	NAMES onnxruntime_cxx_api.h
	HINTS
		${ONNXRUNTIME_ROOT_DIR}/include
		$ENV{ONNXRUNTIME_ROOT_DIR}/include
	PATHS
		${ONNXRUNTIME_WC_INCLUDES}
		/usr/include
		/usr/local/include
)

find_library(OnnxRuntime_LIBRARY
	NAMES onnxruntime
	HINTS
		${ONNXRUNTIME_ROOT_DIR}/lib
		$ENV{ONNXRUNTIME_ROOT_DIR}/lib
	PATHS
		${ONNXRUNTIME_WC_LIBS}
		/usr/lib
		/usr/local/lib
)

find_package_handle_standard_args(OnnxRuntime REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
        IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)

