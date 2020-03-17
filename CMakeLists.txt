cmake_minimum_required (VERSION 3.5)
project(test_opencv)

set (CMAKE_CXX_FLAGS "-std=c++11 -fPIC")

# Opencv
find_package(OpenCV REQUIRED)
message(STATUS "OPENCV_INCLUDE_DIRS" ${OppenCV_INCLUDE_DIRS})
add_executable(test_cv test_opencv.cpp)
target_link_libraries(test_cv ${OpenCV_LIBS}) 

# find cuda
find_package(CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
message (STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

file(GLOB HEADERS *.h)
file(GLOB SOURCES *.cpp *.cu)
cuda_add_library(GPU_NORM STATIC ${HEADERS} ${SOURCES})

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
add_executable(test_normalize normalize_test.cpp)
target_link_libraries(test_normalize GPU_NORM ${OpenCV_LIBS})
