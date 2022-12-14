cmake_minimum_required(VERSION 3.23)
project(tsp C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})


add_executable(v1 v1.c)
target_include_directories(v1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1 ${OpenCL_LIBRARY})

add_executable(v1.1 v1.1.c)
target_include_directories(v1.1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.1 ${OpenCL_LIBRARY})

add_executable(v1.1.1 v1.1.1.c)
target_include_directories(v1.1.1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.1.1 ${OpenCL_LIBRARY})

add_executable(v1.1.2 v1.1.2.c)
target_include_directories(v1.1.2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.1.2 ${OpenCL_LIBRARY})

add_executable(v1.1.3 v1.1.3.c)
target_include_directories(v1.1.3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.1.3 ${OpenCL_LIBRARY})

add_executable(v1.2 v1.2.c)
target_include_directories(v1.2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.2 ${OpenCL_LIBRARY})

add_executable(v1.2.1 v1.2.1.c)
target_include_directories(v1.2.1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v1.2.1 ${OpenCL_LIBRARY})

add_executable(v2 v2.c)
target_include_directories(v2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2 ${OpenCL_LIBRARY})

add_executable(v2.1 v2.1.c)
target_include_directories(v2.1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2.1 ${OpenCL_LIBRARY})

add_executable(v2.2 v2.2.c)
target_include_directories(v2.2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2.2 ${OpenCL_LIBRARY})

add_executable(v3 v3.c)
target_include_directories(v3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v3 ${OpenCL_LIBRARY})


add_executable(list list.c)
target_include_directories(list PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(list ${OpenCL_LIBRARY})

add_executable(main tsp.c)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main ${OpenCL_LIBRARY})

add_executable(test test.c)