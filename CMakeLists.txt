cmake_minimum_required(VERSION 3.20)
project(v4q128 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library target
add_library(v4q128 INTERFACE)
target_include_directories(v4q128 INTERFACE include)

# Compiler target flags
if(MSVC)
    target_compile_options(v4q128 INTERFACE /arch:AVX2 /O2)
else()
    target_compile_options(v4q128 INTERFACE -mavx2 -mfma -O3 -Wall -Wextra -Wpedantic)
endif()

# Development sandbox executable
add_executable(sandbox sandbox/main.cpp)
target_link_libraries(sandbox PRIVATE v4q128)
