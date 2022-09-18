cmake_minimum_required(VERSION 3.23)
project(tsp C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(Ech_AMga_CApp_MCcpu Ech_AMga_CApp_MCcpu.c)

target_include_directories(Ech_AMga_CApp_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech_AMga_CApp_MCcpu ${OpenCL_LIBRARY})

add_executable(list list.c)

target_include_directories(list PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(list ${OpenCL_LIBRARY})

add_executable(bruteforce sequential.c)

add_executable(test test.c)