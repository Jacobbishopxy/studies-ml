# Studies PyTorch

## Installation

1. Visit <https://pytorch.org/> and select a suitable libtorch version to download.

    For example:

    ```sh
    # If you need e.g. CUDA 9.0 support, please replace "cpu" with "cu90" in the URL below.
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip
    ```

1. Place the unzipped `libtorch` folder under the root of the project directory，and pick either one below

    ```txt
    # list(APPEND CMAKE_PREFIX_PATH "libtorch")
    list(APPEND Torch_DIR, "libtorch")
    ```

    ahead of `find_package(Torch REQUIRED)` in `CMakeLists.txt`.

*Note:* For Mac M1 user, besides following the steps above, we'll have several extra steps to be done. Check [this](./mac_m1_build.md).

## Study materials

1. Official website: <https://pytorch.org/>

1. [PyTorch Cpp](https://github.com/prabhuomkar/pytorch-cpp) and [PyTorch Py](https://github.com/yunjey/pytorch-tutorial)

## Contents

1. Starter

    - [Pytorch C++ API](./starter/introduction.cpp)

    - [Using the PyTorch C++ Frontend](./starter/using_cpp_frontend.cpp)

1. Basics

1. Intermediate

1. Advanced

1. Interactive
