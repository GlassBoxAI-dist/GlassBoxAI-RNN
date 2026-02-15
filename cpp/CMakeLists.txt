cmake_minimum_required(VERSION 3.14)
project(facaded_rnn_cpp VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Rust library location (set after cargo build)
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release" CACHE PATH "Path to Rust built libraries")

add_library(facaded_rnn_c INTERFACE)
target_include_directories(facaded_rnn_c INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_library(facaded_rnn_cpp INTERFACE)
target_include_directories(facaded_rnn_cpp INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(facaded_rnn_cpp INTERFACE facaded_rnn_c)

# Example binary
add_executable(example example.cpp)
target_link_libraries(example PRIVATE facaded_rnn_cpp)
target_link_directories(example PRIVATE ${RUST_LIB_DIR})
target_link_libraries(example PRIVATE facaded_rnn_cpp_lib)

if(UNIX AND NOT APPLE)
    target_link_libraries(example PRIVATE dl pthread m)
endif()
if(APPLE)
    target_link_libraries(example PRIVATE dl pthread)
endif()

# The actual Rust static/shared lib name
# After `cargo build --release`, link against libfacaded_rnn_cpp.a or .so
add_library(facaded_rnn_cpp_lib STATIC IMPORTED)
set_target_properties(facaded_rnn_cpp_lib PROPERTIES
    IMPORTED_LOCATION "${RUST_LIB_DIR}/libfacaded_rnn_cpp.a"
)
