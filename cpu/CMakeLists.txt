cmake_minimum_required(VERSION 3.6)
project(cpu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -Wno-write-strings")

set(SOURCE_FILES
    src/main.cpp)

add_executable(cpu ${SOURCE_FILES})
target_link_libraries(cpu fftw3 sndfile)
