# Building libtorch using CMake

**IMPORTANT!!** do not use homebrew to install `pytorch`, otherwise you will get a "broken" lib which later throws errors like `Undefined symbols for architecture arm64 ...`.

**IMPORTANT!!** please use Clang as your compiler, since `libtorch` is also built on it.

Source: <https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst>

And stack overflow's answer: <https://stackoverflow.com/a/69947763>

```sh
# clone source file and name it `libtorch_builder` ()
git clone -b master --recurse-submodule https://github.com/pytorch/pytorch.git libtorch_builder

# create a new directory to hold the future compiled files
mkdir pytorch-build
cd pytorch-build

# build from source code `libtorch_builder` and place the compiled files to `pytorch-install`
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../libtorch_builder

# install
cmake --build . --target install

# copy the installed files to the required place (which in this project is `/libtorch`)
cd ..
cp -r pytorch-install/** ./studies-pytorch/libtorch

# build project
cd studies-pytorch/build
cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make
```
