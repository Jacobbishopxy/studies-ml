cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

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

if(DOWNLOAD_DATASETS)
    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})

    # mnist
    add_custom_target(
        mnist COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_mnist.cmake
    )
    # imagenette
    add_custom_target(
        imagenette COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/imagenette.cmake
    )
endif()

add_subdirectory(./starter st)

add_subdirectory(./basics bs)
add_subdirectory(./intermediate im)
add_subdirectory(./advanced ad)
add_subdirectory(./interactive ia)

add_subdirectory("utils/image_io")
