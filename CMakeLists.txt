cmake_minimum_required(VERSION 3.14)
project(VehicleCounter_V2)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

add_executable(VehicleCounter_V2 main.cpp Blob.cpp Blob.h)

target_link_libraries( VehicleCounter_V2 ${OpenCV_LIBRARIES} )

install (TARGETS VehicleCounter_V2 DESTINATION bin)