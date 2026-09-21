cmake_minimum_required(VERSION 3.20)
project(v4q128 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to toggle test building
option(V4Q128_BUILD_TESTS "Build test suite for v4q128" ON)

# Define all library header files so IDEs index them
set(V4Q128_HEADERS
    include/v4q128/v4q128.hpp
    include/v4q128/core/storage.hpp
    include/v4q128/io/transpose.hpp
    include/v4q128/io/add_sub.hpp
    include/v4q128/io/cmp.hpp
    include/v4q128/ops/mul.hpp
)

# Header-only library target
add_library(v4q128 INTERFACE)
target_include_directories(v4q128 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(v4q128 INTERFACE ${V4Q128_HEADERS})

# Compiler target flags
if(MSVC)
    target_compile_options(v4q128 INTERFACE /arch:AVX2 /O2)
else()
    target_compile_options(v4q128 INTERFACE -mavx2 -mfma -O3 -Wall -Wextra -Wpedantic)
endif()

# Development sandbox executable
add_executable(sandbox sandbox/main.cpp ${V4Q128_HEADERS})
target_link_libraries(sandbox PRIVATE v4q128)

# CTest / Test Suite Setup
if(V4Q128_BUILD_TESTS)
    enable_testing()

    add_executable(test_mul tests/test_mul.cpp ${V4Q128_HEADERS})
    target_link_libraries(test_mul PRIVATE v4q128)

    add_test(NAME MultiplicationStressTest COMMAND test_mul)
endif()
