# Studies ML

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

## Contents

Note: most of the comments and `.tex` files are written in Chinese [ch-CN].

### Notes

1. Studies of the *Watermelon* Book: [Main entry](./notes-watermelon.tex) & [PDF](./out/notes-watermelon.pdf)

    - [Model evaluation and selection](./notes-watermelon/model_evaluation_and_selection.tex)

    - [Linear model](./notes-watermelon/linear_model.tex)

    - Decision tree

    - [Neural networks](./notes-watermelon/neural_networks.tex)

1. Studies of *Essential Math for AI*: [Main entry](./notes-emfa.tex) & [PDF](./out/notes-emfa.pdf).

    - Data

    - Singular value decomposition

### Codes

1. Starter

    - [Pytorch C++ API](./starter/introduction.cpp)

    - [Using the PyTorch C++ Frontend](./starter/using_cpp_frontend.cpp)

1. Basics

    - basics: [cpp](./basics/basics.cpp) & [py](./basics/basics.py)

    - linear regression: [cpp](./basics/linear_regression.cpp) & [py](./basics/linear_regression.py)

    - logistic regression: [cpp](./basics//logistic_regression.cpp) & [py](./basics/logistic_regression.cpp)

    - feed forward neural net: [cpp](./basics/feed_forward_neural_network.cpp) & [py](./basics/feed_forward_neural_network.py)

1. Intermediate

    - convolutional neural network: [cpp](./intermediate/convolutional_neural_network/convolutional_neural_network.cpp) & [py](./intermediate/convolutional_neural_network/convolutional_neural_network.py)

    - deep residual network: [cpp](./intermediate/deep_residual_network/deep_residual_network.cpp) & [py](./intermediate/deep_residual_network/deep_residual_network.py)

1. Advanced

1. Interactive

## Project Structure

### CMake deps

- extern

- utils

## References

1. WatermelonBook [ch-CN]: <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm>

1. PumpkinBook [ch-CN]: <https://github.com/datawhalechina/pumpkin-book>

1. Essential Math for AI [en-US]: <https://www.oreilly.com/library/view/essential-math-for/9781098107628/>

1. PyTorch: <https://pytorch.org/>

1. [PyTorch Cpp](https://github.com/prabhuomkar/pytorch-cpp) and [PyTorch Py](https://github.com/yunjey/pytorch-tutorial)
