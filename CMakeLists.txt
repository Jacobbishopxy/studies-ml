cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(studies-ml LANGUAGES CXX VERSION 0.0.1)

list(APPEND CMAKE_PREFIX_PATH "libtorch")
set(Torch_DIR, "libtorch")

# find_package(Torch REQUIRED)
find_package(Torch QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# datasets download
option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." OFF)
add_definitions(-D_DATASETS_PATH=\"${CMAKE_SOURCE_DIR}/data\")

add_subdirectory(./starter st)

add_subdirectory(./basics bs)
add_subdirectory(./intermediate im)
add_subdirectory(./advanced ad)
add_subdirectory(./interactive ia)
