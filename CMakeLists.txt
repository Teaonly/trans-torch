CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA)

# Caffe may have been compiled without CUDA support
# (Caffe headers check for CPU_ONLY flag)
IF(NOT CUDA_FOUND)
	add_definitions(-DCPU_ONLY)
	SET(CUDA_INCLUDE_DIRS "")
ENDIF()

IF(NOT EXISTS ${CAFFE_DIR})
  SET(CAFFE_DIR "$ENV{CAFFE_DIR}")
ENDIF()
MESSAGE("CAFFE_DIR: " ${CAFFE_DIR})

# Any thrid include and library path can be set here
SET(OPT_DIR "~/opt")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH" "${CAFFE_DIR}/include" "${CAFFE_DIR}/build/src" ${CUDA_INCLUDE_DIRS} "${OPT_DIR}/include")

LINK_DIRECTORIES("${Torch_INSTALL_LIB}" "${CAFFE_DIR}/build/lib")
add_definitions(-Wl,-as-needed)

SET(src caffe.cpp ${PROTO_SRCS})

FILE(GLOB luasrc *.lua)

ADD_LIBRARY(trans_torch MODULE ${src} "${OPT_DIR}/lib")
TARGET_LINK_LIBRARIES(trans_torch TH caffe)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(trans_torch PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
    
INSTALL(TARGETS trans_torch
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/transtorch")
