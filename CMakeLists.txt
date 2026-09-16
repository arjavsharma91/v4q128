cmake_minimum_required(VERSION 3.20)
project(v4q128 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define all library header files so IDEs index them
set(V4Q128_HEADERS
        include/v4q128/v4q128.hpp
        include/v4q128/core/storage.hpp
        include/v4q128/io/transpose.hpp
        include/v4q128/io/add_sub.hpp
)

# Header-only library target
add_library(v4q128 INTERFACE)
target_include_directories(v4q128 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(v4q128 INTERFACE ${V4Q128_HEADERS})

# Compiler target flags
if(MSVC)
    target_compile_options(v4q128 INTERFACE /arch:AVX2 /O2)
else()
    target_compile_options(v4q128 INTERFACE -march=native -O3 -Wall -Wextra -Wpedantic)
endif()

# Development sandbox executable
add_executable(sandbox sandbox/main.cpp ${V4Q128_HEADERS})
target_link_libraries(sandbox PRIVATE v4q128)
