cmake_minimum_required(VERSION 3.23)
project(tsp C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})


add_executable(Ech1_AMga_CAsr_MCcpu Ech_ViIobrCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech1_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech1_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})

add_executable(Ech2_AMga_CAsr_MCcpu Ech_VcIobrCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech2_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech2_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})

add_executable(Ech3_AMga_CAsr_MCcpu Ech_ViIobcwCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech3_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech3_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})

add_executable(Ech4_AMga_CAsr_MCcpu Ech_VcIobcwCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech4_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech4_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})

add_executable(Ech5_AMga_CAsr_MCcpu Ech_ViIobcmCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech5_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech5_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})

add_executable(Ech6_AMga_CAsr_MCcpu Ech_VcIobcmCbr_AMga_CAsr_MCcpu.c)
target_include_directories(Ech6_AMga_CAsr_MCcpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Ech6_AMga_CAsr_MCcpu ${OpenCL_LIBRARY})


add_executable(list list.c)
target_include_directories(list PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(list ${OpenCL_LIBRARY})

add_executable(bruteforce bruteforce.c)

add_executable(test test.c)