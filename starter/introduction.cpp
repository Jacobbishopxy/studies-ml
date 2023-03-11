/**
 * @file:	introduction.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/07 08:58:55 Tuesday
 * @brief:
 *
 * https://pytorch.org/cppdocs/
 * 1. ATen
 * 2. Autograd
 * 3. C++ Frontend
 * 4. TorchScript
 * 5. C++ Extensions
 **/

#include <torch/torch.h>

/*
2023/03/08

PyTorch C++ API

- ATen: 基础张量 tensor 以及数学运算库都基于此
- Autograd: 带有自动微分的 ATen 增强
- C++ Frontend: 机器学习中训练以及评估模型的高等级构造器
- TorchScript: TorchScript JIT 编译器以及解释器的接口
- C++ Extensions: 通过自定义的 C++ 以及 CUDA 相关拓展 Python API
*/

// ================================================================================================
// ATen
//
// 文档：https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at
//
// ATen 是一个张量库的基础，PyTorch 中几乎所有其它的 Python 以及 C++ 接口都是基于它所构建。它提供了一个核心的
// `Tensor` 类，其之上被定义了数百的操作。大多数这些操作同时都拥有 CPU 以及 GPU 的实现，其中根据 `Tensor` 类
// 本身的类型来动态的进行调度。
// ================================================================================================

#include <ATen/ATen.h>

void intro_aten()
{

  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});

  auto c = a + b.to(at::kInt);
}

// ================================================================================================
// Autograd
//
// 我们所说的 autograd 是 PyTorch 的 C++ API 的一部分，它们通过有关自动微分的功能来增强 ATen `Tensor` 类。
// autograd 系统记录对张量的操作以形成 autograd 图。在此图中叶变量上调用 `backwards()`，通过 autograd 图
// 的函数和张量网络执行反向模式微分，最终产生梯度。
// ================================================================================================

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

void intro_autograd()
{
  torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
  torch::Tensor b = torch::randn({2, 2});

  auto c = a + b;

  c.backward(); // a.grad() 将会存储 c 的 w.r.t. a. 的梯度
}

// ================================================================================================
// ATen 中的 `at::Tensor` 类默认是不可微分的。为了通过 autograd API 给张量添加可微，用户必须从 `torch::` 命名
// 空间中使用张量工厂函数。例如，`at::ones` 所创造的一个张量是不可微的，而 `torch::ones` 则可微。
// ================================================================================================

// ================================================================================================
// C++ Frontend
//
// PyTorch C++ 端提供了高等级，纯 C++ 的神经网络与通用 ML（机器学习） 研究与生产使用的建模接口，很大程度上遵循了
// Python 的 API 设计以及提供的功能。C++ 端包含了以下：
// - 一个通过层级模块系统（例如 `torch::nn::Module`）来定义机器学习模型的接口；
// - 一个有着最常用建模目的（例，卷积，RNN，批标准化等）的模块“标准库”；
// - 一个优化 API，包含了常用的优化器实现，例如 SGD，Adam，RMSprop 等等；
// - 一种表示数据集与数据管道的方法，包括通过很多 CPU 核来进行的并行加载数据功能
// - 一种序列化格式，用于存储和加载训练会话的检查点（例如 `torch.utils.data.DataLoader`）；
// - 自动并行化模型到多个 GPUs（例如 `torch.nn.parallel.DataParallel`）；
// - 通过 pybind11，支持代码更容易得绑定 C++ 模型至 Python
// - TorchScript JIT 编译器的入口
// - 功能性的工具提升 ATen 以及 Autograd APIs
//
// 更详细的说明请参见文档：https://pytorch.org/cppdocs/frontend.html
// 更多的例子请参见项目：https://github.com/pytorch/examples/tree/master/cpp
//
// ================================================================================================

// ================================================================================================
// TorchScript
//
// TorchScript 是 PyTorch 模型的一种展示，其可以被 TorchScript 编译器所理解，编译并且序列化。根本上来说，
// TorchScript 是一个编程语言，它是使用了 PyTorch API 的 Python 子集。C++ 对 TorchScript 的接口包含了三种
// 主要的功能：
// - 一种用于加载和执行 Python 中定义的序列化模型的机制；
// - 定义自定义操作器的 API，用于拓展 TorchScript 标准库的操作器
// - C++ 所带来的恰到好处的 TorchScript 程序编译
//
// 更多文档：https://pytorch.org/tutorials/advanced/cpp_export.html
// ================================================================================================

// ================================================================================================
// C++ Extensions
//
// C++ 扩展提供了一种简单而强大的方法来访问上述所有接口，以扩展 PyTorch 的常规 Python 用例。C++ 扩展最常用于在
// C++ 或 CUDA 中实现自定义运算符，以加速对普通 PyTorch 的研究。C++ 扩展 API 不会向 PyTorch C++ API 添加任何
// 新功能。相反，它提供与 Python setuptools 的集成以及允许从 Python 访问 ATen、autograd 和其他 C++ API 的
// JIT 编译机制。
//
// 更多请访问 Tutorial：https://pytorch.org/tutorials/advanced/cpp_extension.html
// ================================================================================================

int main()
{
}
