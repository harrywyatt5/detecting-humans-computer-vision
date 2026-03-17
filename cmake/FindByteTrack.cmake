# Is a helper library for finding libraries on the system
include(FindPackageHandleStandardArgs)

find_path(ByteTrack_INCLUDE_DIR
	NAMES ByteTrack/BYTETracker.h
	HINTS
        ${BYTETRACK_ROOT_DIR}/build
		${BYTETRACK_ROOT_DIR}/include
        $ENV{BYTETRACK_ROOT_DIR}/build
		$ENV{BYTETRACK_ROOT_DIR}/include
	PATHS
		/opt/ByteTrack-cpp/include
        /usr/local/ByteTrack-cpp/include
        "$ENV{HOME}/ByteTrack-cpp/include"
		/usr/include
		/usr/local/include
)

find_library(ByteTrack_LIBRARY
	NAMES bytetrack
	HINTS
		${BYTETRACK_ROOT_DIR}/lib
		$ENV{BYTETRACK_ROOT_DIR}/lib
	PATHS
		/opt/ByteTrack-cpp/build
        /usr/local/ByteTrack-cpp/lib
        "$ENV{HOME}/ByteTrack-cpp/lib"
		/usr/lib
		/usr/local/lib
)

find_package_handle_standard_args(ByteTrack REQUIRED_VARS ByteTrack_LIBRARY ByteTrack_INCLUDE_DIR)

if(ByteTrack_FOUND AND NOT TARGET ByteTrack::ByteTrack)
    add_library(ByteTrack::ByteTrack UNKNOWN IMPORTED)
    set_target_properties(ByteTrack::ByteTrack PROPERTIES
        IMPORTED_LOCATION "${ByteTrack_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ByteTrack_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(ByteTrack_INCLUDE_DIR ByteTrack_LIBRARY)
