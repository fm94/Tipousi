cmake_minimum_required(VERSION 3.0.0)
project(Tipousi VERSION 0.1.0)
 
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Activations
    ${PROJECT_SOURCE_DIR}/include/Data
    ${PROJECT_SOURCE_DIR}/include/Layers
    ${PROJECT_SOURCE_DIR}/include/Losses
    ${PROJECT_SOURCE_DIR}/include/Metrics
    ${PROJECT_SOURCE_DIR}/include/Module
    ${PROJECT_SOURCE_DIR}/include/Sequential
    ${PROJECT_SOURCE_DIR}/include/Trainers
    ${PROJECT_SOURCE_DIR}/src/Trainers
)

# source files
file(GLOB SOURCES "src/*/*.cpp" tests/main.cpp)

include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")

add_executable(Tipousi ${SOURCES})
