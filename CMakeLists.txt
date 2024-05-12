cmake_minimum_required(VERSION 3.0.0)
project(Tipousi VERSION 0.0.1)
 
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/activation
    ${PROJECT_SOURCE_DIR}/include/loss
)

# source files
file(GLOB SOURCES "src/*/*.cpp" tests/main.cpp)

include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")

add_executable(Tipousi ${SOURCES})
