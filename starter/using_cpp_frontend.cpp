/**
 * @file:	define_nn_models.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/07 15:13:54 Tuesday
 * @brief:
 *
 * 教程：https://pytorch.org/tutorials/advanced/cpp_frontend.html
 *
 * 1. 模块 API 基础
 * 2. 定义 DCCGAN 模块
 **/

#include <iostream>
#include <torch/torch.h>

// ================================================================================================
// 定义神经网络模型
// ================================================================================================

// ================================================================================================
// 1. 模块 API 基础
//
// 与 Python 接口一致，C++ 的 nn 是由被称为 modules 的可复用模块所构成（`torch::nn::Module`）。
// 除了一个封装了算法的实现的 `forward()` 方法，一个 module 通常由以下任意的三种子对象所构成：
// 参数 parameters，缓冲区 buffers 以及子模块 submodules。
//
// parameters 与 buffers 存储了张量 tensors 的状态。前者记录梯度 gradients，后者则不。parameters 通常是
// 神经网络中的可训练全职 trainable weights；而 buffers 包含了例如 batch 标准化的 means 以及 variances。
// 为了复用特定的逻辑块或者状态，PyTorch API 允许嵌套模块，而嵌套模块则被视为子模块。
//
// Parameters，buffers 与 submodules 必须被显式的注册。一旦注册，像是 `parameters()` 或者 `buffers()`
// 等方法则可以被用于获取一个容器中所有的 parameters（包括嵌套中的）。同样的，像是 `to(...)` 这样的方法，
// 例如 `to(torch::kCUDA)` 将会从 CPU 中移动所有 parameters 与 buffers 至 CUDA 内存，即有层级的作用于
// 所有的模块。
// ================================================================================================

// ================================================================================================
// 定义一个模块 Module 并注册参数 Parameters
// ================================================================================================

struct Net1 : torch::nn::Module
{
  Net1(int64_t N, int64_t M)
  {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input)
  {
    return torch::addmm(b, input, W);
  }

  torch::Tensor W, b;
};

// ================================================================================================
// 注册子模块以及遍历模块层级
//
// 注册了 parameters，那么注册 submodules 也是可以的。在 C++ 中，使用恰当的 `register_module()` 方法
// 用于注册一个 `torch::nn::Linear`
//
// 可以在这里 https://pytorch.org/cppdocs/api/namespace_torch__nn.html 找到位于 `torch::nn` 命名空间下的
// 所有内建 modules。
// ================================================================================================

struct Net2 : torch::nn::Module
{
  Net2(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M)))
  {
    another_bias = register_parameter("b", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input)
  {
    return linear(input) + another_bias;
  }

  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

// ================================================================================================
// 上述代码中的细微差别在于：子模块是在构造函数的初始化列表中构造，而参数则是在构造函数体内构造。
// 这里的好处是在于之后会提到的 C++ 版本的 ownership model。
// ================================================================================================

// ================================================================================================
// 调用 `parameters()` 将返回一个 `std::vector<torch::Tensor>`。下面对其进行遍历：
// ================================================================================================

void print1()
{
  Net1 net(4, 5);

  std::cout << "print1: " << std::endl;
  for (const auto& p : net.parameters())
  {
    std::cout << p << std::endl;
  }
}

void print2()
{
  Net2 net(4, 5);

  std::cout << "print2: " << std::endl;
  for (const auto& p : net.parameters())
  {
    std::cout << p << std::endl;
  }
}

// ================================================================================================
// 也可以使用 C++ 提供的 `named_parameters()` 方法来返回类似 Python 那样的 `OrderedDict` 结构.
//
// 注：文档 https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module
// 包含了所有 `torch::nn::Module` 操作 module 的方法。
// ================================================================================================

void print3()
{
  Net2 net(4, 5);

  std::cout << "print3: " << std::endl;
  for (const auto& pair : net.named_parameters())
  {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
}

// ================================================================================================
// 以前向模式 Forward Mode 运行网络
//
// 在 C++ 中运行网络只需要简单的调用之前定义过的 `forward()` 方法即可：
// ================================================================================================

void forward()
{
  Net2 net(4, 5);
  std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}

// ================================================================================================
// Module Ownership
//
// Ownership model 指代 modules 存储与传递的一种方式 -- 即决定了谁或者什么拥有 owns 一个特定的模块实例
// module instance。在 Python 中，对象总是动态分配的（在堆上）且拥有引用语义。这样工作起来很方便，也便于
// 理解。实际上在 Python 中，用户可以最大程度上的忘记对象的存在，以及对象是如何被引用的，而专注于业务逻辑
// 的本身。
//
// C++ 作为一个低级的语言，在此方面提供了更多的选项。这增加了复杂度以及很大程度上的影响了 C++ 端的设计与
// 人体工学。特别是对于 C++ 端，我们可以选择使用值语义 value semantics 或是引用语义 reference semantics。
// 前者是最简单的，并且上面的例子也都展示过了：module 对象被分配在栈上，同时当传递给一个函数时，可以被拷贝，
// 移动（通过 `std::move`）或者被引用，或者通过指针：
// ================================================================================================

struct Net3 : torch::nn::Module
{
};

void a(Net3 net) {}
void b(Net3& net) {}
void c(Net3* net) {}

void module_ownership_test()
{
  Net3 net;

  a(net);
  a(std::move(net));
  b(net);
  c(&net);
}

// ================================================================================================
// 对于第二个案例 -- 引用语义 -- 我们可以使用 `std::shared_ptr`。而引用语义的优势在于，如同 Python，
// 它减少了思考 modules 该如何传递给函数以及值该如何被声明（假设用户到处使用 `shared_ptr`）的认知开销。
// ================================================================================================

struct Net4 : torch::nn::Module
{
};

void a(std::shared_ptr<Net4> net) {}

void module_ownership_test2()
{
  auto net = std::make_shared<Net4>();

  a(net);
}

// ================================================================================================
// 根据经验，从动态语言而来的研究者更大程度的喜爱引用语义而不是值语义，尽管后者对于 C++ 而言更“自然”。
// 同样重要的是注意 `torch::nn::Module` 的设计，为了更贴近 Python API 的人体工学，它依靠的是共享所有权
// shared ownership。
//
// 拿之前的 `Net2` 作为例子，为了使用 `linear` 子模块，我们希望将其直接存储在类中。然而，我们同样希望模块
// 的基类知晓并拥有此子模块的访问权。为此，基类必须存储该子模块的引用。此时我们已经需求用到共享所有权了。
// `torch::nn::Module` 类与具体 `Net` 类都需要子模块的引用。为此，基类以 `shared_ptr` 来存储模块，同样
// 具体类也需要这么做。
//
// 但是等一下！上述的代码并没有提到任何 `shared_ptr`！这是为什么呢？因为 `std::shared_ptr<MyModule>` 是
// 一个复杂的类型。为了提高研究员的生产力，我们提出了一个周全的计划来隐藏 `shared_ptr` -- 这通常是值语义的
// 好处 -- 同事保留引用语义。为了理解这是如何工作的，简单的来看一下核心库中 `torch::nn::Linear` 的定义，
// 全定义在此：https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h
//
// ```cpp
// struct LinearImpl : torch::nn::Module {
//   LinearImpl(int64_t in, int64_t out);
//   Tensor forward(const Tensor& input);
//   Tensor weight, bias;
// };
// TORCH_MODULE(Linear);
// ```
//
// 简单来说，模块并非被称为 `Linear`，而是 `LinearImpl`。一个宏，`TORCH_MODULE` 才是真正定义了 `Linear`
// 类。这个 “被生成的” 类实际上是一个 `std::shared_ptr<LinearImpl>` 的包装器。正因其并非是一个简单的
// typedef，因此构造函数仍然能如预期那样工作。例如，可以这么写 `torch::nn::Linear(3,4)` 而不是
// `std::make_shared<LinearImpl>(3,4)`。我们将由宏所创建的模块称为 holder。类似于（shared）pointers，
// 用户可以使用箭头符号访问对象（即 `model->forward(...)`）。这样最终的结果就是 ownership model 的构造
// 与 Python API 非常的相似。引用语义成为了默认情况，而不再语言额外的类型 `std::shared_ptr` 或者是
// `std::make_shared`。对于我们的 `Net`，使用模块 holder 的 API 就像是这样：
//
// ```cpp
// struct NetImpl : torch::nn::Module {};
// TORCH_MODULE(Net);
// void a(Net net) { }
// int main() {
//   Net net;
//   a(net);
// }
// ```
//
// 这里有一个值得被提到的微小问题。默认构造的 `std::shared_ptr` 是 “空的”，也就是包含一个空指针。那么默认
// 构造的 `Linear` 或者 `Net` 是什么呢？好吧，这是一个棘手的选择。我们可以说它应该是一个空（null）的
// `std::shared_ptr<LinearImpl>`；然而，调用 `Linear(3,4)` 就与 `std::make_shared<LinearImpl>(3,4)`
// 一样了。这就意味着如果我们决定了 `Linear linear;` 就必须是一个空指针，那么这样就没有办法构建一个
// 构造函数不带有任何参数的模块，或者是构造函数的参数全是默认的情况。正因这个原因，现有的 API，一个默认
// 构造的模块 holder（像是 `Linear()`）将唤起其基础模块的默认构造函数（`LinearImpl()`）。如果基础模块
// 没有默认构造函数，则会报编译错误。为了不去构造空的 holder，用户可以传递一个 `nullptr` 进 holder 的
// 构造函数。
//
// 实践中，这就意味着用户可以尽早的使用子模块（初始化列表），即 `Net2`；或者是在 holder 的构造函数中放入
// `nullptr`（更符合 Pythonistas 了）：
// ================================================================================================

struct Net5 : torch::nn::Module
{
  Net5(int64_t N, int64_t M)
  {
    linear = register_module("linear", torch::nn::Linear(N, M));
  }
  torch::nn::Linear linear{nullptr}; // construct an empty holder
};

// ================================================================================================
// 2. 定义 DCCGAN 模块
// ================================================================================================

// TODO

int main(int argc, char** argv)
{

  return 0;
}
