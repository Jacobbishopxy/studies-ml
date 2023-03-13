/**
 * @file:	using_cpp_frontend.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/07 15:13:54 Tuesday
 * @brief:
 *
 * 教程：https://pytorch.org/tutorials/advanced/cpp_frontend.html
 *
 * 1. 定义神经网络模型
 *    - 模块 API 基础
 *    - 定义 DCCGAN 模块
 * 2. 加载数据
 * 3. 编写训练循环
 * 4. 移动模型至 GPU
 * 5. 记录点与恢复训练
 * 6. 检查生成的图像
 **/

#include <iostream>
#include <torch/torch.h>

using namespace torch;

// ================================================================================================
// 1. 定义神经网络模型
// ================================================================================================

// ================================================================================================
// a) 模块 API 基础
//
// 与 Python 接口一致，C++ 的 nn 是由被称为 modules 的可复用模块所构成（`nn::Module`）。
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

struct Net1 : nn::Module
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
// 用于注册一个 `nn::Linear`
//
// 可以在这里 https://pytorch.org/cppdocs/api/namespace_torch__nn.html 找到位于 `nn` 命名空间下的
// 所有内建 modules。
// ================================================================================================

struct Net2 : nn::Module
{
  Net2(int64_t N, int64_t M) : linear(register_module("linear", nn::Linear(N, M)))
  {
    another_bias = register_parameter("b", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input)
  {
    return linear(input) + another_bias;
  }

  nn::Linear linear;
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
// 包含了所有 `nn::Module` 操作 module 的方法。
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

struct Net3 : nn::Module
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

struct Net4 : nn::Module
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
// 同样重要的是注意 `nn::Module` 的设计，为了更贴近 Python API 的人体工学，它依靠的是共享所有权
// shared wonership。
//
// 拿之前的 `Net2` 作为例子，为了使用 `linear` 子模块，我们希望将其直接存储在类中。然而，我们同样希望模块
// 的基类知晓并拥有此子模块的访问权。为此，基类必须存储该子模块的引用。此时我们已经需求用到共享所有权了。
// `nn::Module` 类与具体 `Net` 类都需要子模块的引用。为此，基类以 `shared_ptr` 来存储模块，同样
// 具体类也需要这么做。
//
// 但是等一下！上述的代码并没有提到任何 `shared_ptr`！这是为什么呢？因为 `std::shared_ptr<MyModule>` 是
// 一个复杂的类型。为了提高研究员的生产力，我们提出了一个周全的计划来隐藏 `shared_ptr` -- 这通常是值语义的
// 好处 -- 同事保留引用语义。为了理解这是如何工作的，简单的来看一下核心库中 `nn::Linear` 的定义，
// 全定义在此：https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h
//
// ```cpp
// struct LinearImpl : nn::Module {
//   LinearImpl(int64_t in, int64_t out);
//   Tensor forward(const Tensor& input);
//   Tensor weight, bias;
// };
// TORCH_MODULE(Linear);
// ```
//
// 简单来说，模块并非被称为 `Linear`，而是 `LinearImpl`。一个宏，`TORCH_MODULE` 才是真正定义了 `Linear`
// 类。这个 “被生成的” 类实际上是一个 `std::shared_ptr<LinearImpl>` 的包装器。正因其并非是一个简单的
// typedef，因此构造函数仍然能如预期那样工作。例如，可以这么写 `nn::Linear(3,4)` 而不是
// `std::make_shared<LinearImpl>(3,4)`。我们将由宏所创建的模块称为 holder。类似于（shared）pointers，
// 用户可以使用箭头符号访问对象（即 `model->forwar(...)`）。这样最终的结果就是 ownership model 的构造
// 与 Python API 非常的相似。引用语义成为了默认情况，而不再语言额外的类型 `std::shared_ptr` 或者是
// `std::make_shared`。对于我们的 `Net`，使用模块 holder 的 API 就像是这样：
//
// ```cpp
// struct NetImpl : nn::Module {};
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

struct Net5 : nn::Module
{
  Net5(int64_t N, int64_t M)
  {
    linear = register_module("linear", nn::Linear(N, M));
  }
  nn::Linear linear{nullptr}; // construct an empty holder
};

// ================================================================================================
// b) 定义 DCCGAN 模块
//
// 代码地址：https://github.com/pytorch/examples/blob/main/cpp/dcgan/dcgan.cpp
//
// 什么是一个 GAN？
// 一个 GAN 由两个不同的神经网络模型所组成：一个生成器 generator 以及一个鉴别器 discriminator。
// 生成器从一个噪音分布中获取样本，其目的是转换每个噪音样本成一个目标分布的图片 -- 我们以 MNIST 数据做例子。
// 而鉴别器则是从 MNIST 数据中获取真实 real 图片，或者是从生成器中获取假 fake 图片。也就是说，鉴别器用于
// 判断一个图片与真实 real（接近 1）或者假（接近 0）的差距。来自鉴别器的关于生成器所生成的图像真实程度的
// 反馈用作于训练生成器。理论而言，生成器和鉴别器之间的平衡使它们协同改进，使得生成器从目标分布中生成难以
// 区分的图片，欺骗鉴别器使得对于真实与虚假的概率达到 `0.5`。对于用户而言，最终的结果就是机器获取噪音输入
// 以及生成真实图片作为数字输出.
// ================================================================================================

// ================================================================================================
// 生成器 Generator 模块
//
// 现在开始定义一个生成器模块，由一系列转置的 2D 卷积 convolutions，批标准 batch normalizations 以及
// ReLU 激活单元所构成。我们显式的调用用户定义的 `forward()` 函数在模块之间传递输入：
// ================================================================================================

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

struct DCGANGeneratorImpl : nn::Module
{
  DCGANGeneratorImpl(int kNoiseSize) : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
                                       batch_norm1(256),
                                       conv2(nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
                                       batch_norm2(128),
                                       conv3(nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
                                       batch_norm3(64),
                                       conv4(nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh(conv4(x));

    return x;
  }

  nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

// ================================================================================================
// 现在可以对 DCGANGenerator 调用 `forward()` 函数来映射噪音样本至图片了。
//
// 对于特地模块的选择，像是 `nn::ConvTransposed2d` 以及 `nn::BatchNorm2d`，遵循之前那样的结构轮廓。
// 至于 `kNoiseSize` 常量则是决定了输入噪音向量的大小，其被设为 `100`。
// ================================================================================================

TORCH_MODULE(DCGANGenerator);
// DCGANGenerator generator(kNoiseSize);

// ================================================================================================
// 鉴别器 Discrimintor 模块
//
// 鉴别器由一系列的卷积，批标准化以及激活函数所构成。然而卷积则不再是转置的卷积，我们使用一个带有 alpha 值
// 0.2 的有漏洞的 ReLU 而不是原生 ReLU。同样的，最后的激活函数变成了一个 Sigmoid，其将值压缩到 0 至 1 的
// 区间。我们接着就可以将该压缩的值解释为鉴别器所赋予图片真实性的概率了。
//
// 构建鉴别器我们将尝试不同的东西：一个序列模块。类似于 Python，PyTorch 在这里为模型提供了两种 API：
// 函数版本，输入将通过成功函数传递（例如生成器的例子）；以及一个更面向对象的版本，即构建一个包含了整个模型
// 作为子模块的序列 Sequential 模块：
// ================================================================================================

nn::Sequential discriminator(
    // Layer 1
    nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 2
    nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(128),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 3
    nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(256),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 4
    nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
    nn::Sigmoid()
);

// ================================================================================================
// 一个序列化 Sequential 模块即简单的函数组合。第一个子模块的输出作为第二个的输入，第三个的输出又变成第四个
// 的输入，以此类推。
// ================================================================================================

// ================================================================================================
// 2. 加载数据
//
// 数据加载器属于 C++ 端的 `data` api，包含在 `torch::data::` 命名空间中。API 包含了一些不同的组件：
// - 数据加载类
// - 定义数据集的 API
// - 定义数据转换 transforms 的 API，其应用于数据集
// - 定义采样器 samplers，其生成用于索引数据集的索引
// - 一个已有的数据集，数据转换以及采样器库
// ================================================================================================

// ================================================================================================
// 作为演示，这里使用 MNIST 数据集。首先我们标准化图片 `Normalize` 使得它们处于 `-1` 至 `+1` 的范围（原先
// 是 `0` 至 `1` 的范围）。其次应用 `Stack` 校对，取得一批张量并沿着第一维度将它们堆叠成一个张量：
// ================================================================================================

// 详见项目根目录的 CMakeLists.txt 定义
// 以下代码将在 main 函数中展示
// std::string path = std::string{_DATASETS_PATH} + "/mnist";
// auto dataset = data::datasets::MNIST(path)
//                    .map(data::transforms::Normalize<>(0.5, 0.5))
//                    .map(data::transforms::Stack<>());

// ================================================================================================
// 注意 MNIST 数据集应该放置在可执行二进制的 `./mnist` 相对路径下，可以根据该脚本来获取 MNIST 数据集：
// https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03
// （位于本层目录下的 `download_mnist_dataset.py`）
//
// 接下来就是创建一个数据加载器并传入上面读取的数据集。使用 `torch::data::make_data_loader` 可以创建一个
// 新的数据加载器，其返回一个正确类型的 `std::unique_ptr`（浙江取决于数据集的类型，取样器的类型以及一些其它
// 的实现细节）：
// ================================================================================================

// auto data_loader = torch::data::make_data_loader(std::move(dataset))

// ================================================================================================
// 数据加载器还有很多选项。可以在该代码中进行详细查阅：
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/dataloader_options.h
// 例如加速数据加载，可以增加 workers 的数量，该默认值为零，意味着使用的是主线程。如果设置 `workers` 至 2，
// 那么将会生成两个线程进行数据的并行加载。我们也可以增加批大小，从默认的 1 至某个合理的值，例如 64
// （`kBatchSize` 的值）。
// ================================================================================================

// const int64_t kBatchSize = 64;
// auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

// ================================================================================================
// 现在可以编写一个循环来批加载数据，现在暂时打印在窗口：
// ================================================================================================

void print_data(auto& data_loader)
{
  for (torch::data::Example<>& batch : *data_loader)
  {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i)
    {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }
}

// ================================================================================================
// 由数据加载器返回的类型是一个 `torch::data::Example`。这个类型是一个简单的结构体，其包含了一个包含数据的
// `data` 字段，以及一个表示标识的 `target` 字段。由于我们对数据集应用了 `Stack` 校对（代码 388 行），
// 因此数据加载器仅返回一个案例；如果之前没有应用校对，数据加载器则会返回 `std::vector<torch::data::Example<>>`
// 也就是每批一个案例。
// ================================================================================================

// ================================================================================================
// 3. 编写训练循环
//
// 现在让我们完成本例子中的算法部分，并且实现生成器与鉴别器之间的优雅舞曲。首先需要创建两个优化器，一个给
// 生成器一个给鉴别器。这里使用的优化器使用的实现是 Adam 算法（https://arxiv.org/pdf/1412.6980.pdf）：
// ================================================================================================

int main(int argc, char** argv)
{
  // ================================================================================================
  // 超参
  // ================================================================================================

  // The size of the noise vector fed to the generator.
  const int64_t kNoiseSize = 100;

  // The batch size for training.
  const int64_t kBatchSize = 64;

  // The number of epochs to train.
  const int64_t kNumberOfEpochs = 30;

  // Where to find the MNIST dataset.
  const char* kDataFolder = _DATASETS_PATH;

  // After how many batches to create a new checkpoint periodically.
  const int64_t kCheckpointEvery = 200;

  // How many images to sample at every checkpoint.
  const int64_t kNumberOfSamplesPerCheckpoint = 10;

  // Set to `true` to restore models and optimizers from previously saved
  // checkpoints.
  const bool kRestoreFromCheckpoint = false;

  // After how many batches to log a new update with the loss value.
  const int64_t kLogInterval = 10;

  // ================================================================================================
  // 设备 & 模型使用
  // ================================================================================================

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA is available! Training on GPU" << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  // 生成器
  DCGANGenerator generator(kNoiseSize);
  generator->to(device);

  // 鉴别器
  nn::Sequential discriminator(
      // Layer 1
      nn::Conv2d(
          nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)
      ),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 2
      nn::Conv2d(
          nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)
      ),
      nn::BatchNorm2d(128),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 3
      nn::Conv2d(
          nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)
      ),
      nn::BatchNorm2d(256),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 4
      nn::Conv2d(
          nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)
      ),
      nn::Sigmoid()
  );
  discriminator->to(device);

  // 与官网所用的 `.beta1` 不同，这里使用 `.betas`。详见：
  // https://github.com/pytorch/pytorch/issues/47351#issuecomment-970297661
  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).betas({0.5, 0.5})
  );

  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas({0.5, 0.5})
  );

  // 从恢复点中读取数据
  if (kRestoreFromCheckpoint)
  {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  // 数据读取，详见前文
  std::string path = std::string{_DATASETS_PATH} + "/mnist";
  auto dataset = data::datasets::MNIST(path)
                     .map(data::transforms::Normalize<>(0.5, 0.5))
                     .map(data::transforms::Stack<>());
  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  // 数据加载，详见前文
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2)
  );

  // ================================================================================================
  // 接下来就是更新训练的循环。我们将在靠外的循环中遍历数据加载器的每个 epoch，接着再写 GAN 训练代码
  // ================================================================================================

  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader)
    {
      // 使用真实图像训练鉴别器
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data;
      torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // 使用假图像训练鉴别器
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // 训练生成器
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();

      // std::printf(
      //     "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
      //     epoch,
      //     kNumberOfEpochs,
      //     ++batch_index,
      //     batches_per_epoch,
      //     d_loss.item<float>(),
      //     g_loss.item<float>()
      // );

      batch_index++;
      if (batch_index % kLogInterval == 0)
      {
        std::printf(
            "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
            static_cast<long>(epoch),
            static_cast<long>(kNumberOfEpochs),
            static_cast<long>(batch_index),
            static_cast<long>(batches_per_epoch),
            d_loss.item<float>(),
            g_loss.item<float>()
        );

        // ================================================================================================
        // 在评估鉴别器之前，我们将参数 parameters 的梯度清零。在计算损失过后，我们调用 `d_loss.backward()`
        // 通过网络反向传递它来计算新的梯度。我们在假图像上持续这个步骤。没有从数据集中使用图像，而是将随机噪音投入
        // 进生成器中来创建假图像。接着传递这些假图像给鉴别器。此时我们希望鉴别器产生低概率，理想接近于零。一旦计算
        // 出了鉴别器对于真实与虚假图像的损失，便可以推进鉴别器的优化器的步数来更新它的参数。
        //
        // 而训练生成器，同样也是先将梯度设为零，接着重新计算在虚假图像上的鉴别器。不过这次我们希望鉴别器所指派的概率接近一，
        // 即表明生成器可以生成以假乱真的图像令鉴别器视为真实（数据集中的图像）。为此我们将 `fake_labels` 的 tensor 全部
        // 设为一。最后将生成器的优化器增加步数并更新其参数。
        // ================================================================================================

        // ================================================================================================
        // 4. 移动模型至 GPU
        //
        // 我们现有的脚本可以完美的运行在 CPU 上，我们都知道卷积运算在 GPU 上会更快。现在让我们快速的看一下如何将训练迁移至
        // GPU 上。我们需要做两件事：将 CPU 设备指定传递给我们分配的 tensors，通过 `to` 方法，显式拷贝任何其它的 tensors
        // 至 GPU。而最简单可以达成上述两者的办法是在训练脚本之前，创建一个 `torch::Device` 实例，然后将该设备传递给 tensor
        // 工厂函数，例如 `torch::zeros`。
        // PS: 由于前文已经做了这一步，本块代码省略。
        // ================================================================================================

        // ================================================================================================
        // 5. 记录点与恢复训练
        //
        // 最后我们应该让我们的训练脚本可以周期性的保存模型参数的状态，优化器的状态，已经一些生成好的图像样本。核心 API 是
        // `torch::save(thing.filename)` 以及 `torch::load(thing.filename)`，这里的 `thing` 可以是一个
        // `torch::nn::Module` 的子类或者是一个优化器的实例，例如本例中的 `Adam` 对象。
        // ================================================================================================

        if (batch_index % kCheckpointEvery == 0)
        {
          // 模型和优化器状态的记录点
          torch::save(generator, "generator-checkpoint.pt");
          torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
          torch::save(discriminator, "discriminator-checkpoint.pt");
          torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");

          // 生成器和保存图像的样本
          torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
          torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
          std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
        }
      }
    }
  }

  // ================================================================================================
  // 6. 检查生成的图像
  //
  // 详见本目录下的 `inspecting_generated_images.py`
  // ================================================================================================

  std::cout << "Training complete!" << std::endl;

  return 0;
}
