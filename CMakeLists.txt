cmake_minimum_required(VERSION 3.10)
project(amdbugdemo_1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})



add_executable(amdbugdemo_1 main.cpp)

target_include_directories (amdbugdemo_1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (amdbugdemo_1 ${OpenCL_LIBRARY})
