cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(studies-pytorch LANGUAGES CXX VERSION 0.0.1)

list(APPEND CMAKE_PREFIX_PATH "libtorch")
set(Torch_DIR, "libtorch")

find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

