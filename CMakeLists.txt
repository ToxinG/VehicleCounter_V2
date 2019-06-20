cmake_minimum_required(VERSION 3.14)
project(VehicleCounter_V2)

set(GCC_COVERAGE_COMPILE_FLAGS "-std=gnu++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

set(PQXX /usr/local/include/pqxx)

find_package(OpenCV REQUIRED)

find_library(PQXX_LIB pqxx)
find_library(PQ_LIB pq)

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

add_executable(VehicleCounter_V2 main.cpp Blob.cpp Blob.h)

target_link_libraries( VehicleCounter_V2 ${OpenCV_LIBRARIES} ${PQXX_LIB} ${PQ_LIB} )

install (TARGETS VehicleCounter_V2 DESTINATION bin)