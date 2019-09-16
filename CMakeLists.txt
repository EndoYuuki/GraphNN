cmake_minimum_required(VERSION 2.8.2)
project(GraphNN)

set(CMAKE_CXX_STANDARD 17) # C++14...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

include_directories(include)
add_executable(GraphNN main.cpp)
target_link_libraries(${PROJECT_NAME} stdc++fs)

# tests
if(NOT without-test)
    enable_testing()
    add_subdirectory(test)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)