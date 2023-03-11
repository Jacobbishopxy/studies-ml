/**
 * @file:	basics.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/10 20:50:12 Friday
 * @brief:
 **/

#include <iomanip>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

// void print_tensor_size(const torch::Tensor&);
void print_script_module(const torch::jit::script::Module& module, size_t spaces = 0);

void print_script_module(const torch::jit::script::Module& module, size_t spaces)
{
  for (const auto& sub_module : module.named_children())
  {
    if (!sub_module.name.empty())
    {
      std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
                << " " << sub_module.name << "\n";
    }

    print_script_module(sub_module.value, spaces + 2);
  }
}

void basic_autograde()
{
  // 创建张量
  torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

  // 构建计算图
  auto y = w * x + b; // y = 2 * x + 3

  // 计算梯度
  y.backward();

  // 打印梯度
  std::cout << x.grad() << std::endl;
  std::cout << w.grad() << std::endl;
  std::cout << b.grad() << std::endl;
}

void basic_autograde2()
{
  // 创建有形张量
  torch::Tensor x = torch::randn({10, 3});
  torch::Tensor y = torch::randn({10, 2});

  // 构建一个全连接层
  torch::nn::Linear linear(3, 2);
  std::cout << "w:\n"
            << linear->weight << std::endl;
  std::cout << "b:\n"
            << linear->bias << std::endl;

  // 创建损失函数与优化器
  torch::nn::MSELoss criterion;
  torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));

  // 前向传递
  auto pred = linear->forward(x);

  // 计算损失
  auto loss = criterion(pred, y);
  std::cout << "Loss: " << loss.item<double>() << std::endl;

  // 反向传递
  loss.backward();

  // 打印梯度
  std::cout << "dL/dw:\n"
            << linear->weight.grad() << std::endl;
  std::cout << "dL/db:\n"
            << linear->bias.grad() << std::endl;

  // 一步的梯度下降
  optimizer.step();

  // 打印一步的梯度下降后的损失
  pred = linear->forward(x);
  loss = criterion(pred, y);
  std::cout << "Loss after 1 optimization step: " << loss.item<double>() << std::endl;
}

void create_tensors_from_existing_data()
{
  // 警告！由 torch::from_blob(ptr_to_data, ...) 创建的 tensors 不会拥有 ptr_to_data 所指向的内存！
  // 详见：https://pytorch.org/cppdocs/notes/tensor_basics.html#using-externally-created-data
  //
  // 如果希望一个 tensor 能拥有数据的拷贝，那么可以在 torch::from_blob() 返回的 tensor 上调用 clone()
  // 例如：
  // torch::Tensor t = torch::from_blob(ptr_to_data, ...).clone();
  // 详解：https://github.com/pytorch/pytorch/issues/12506#issuecomment-429573396

  // 由 C-style 数组构建
  float data_array[] = {1, 2, 3, 4};
  torch::Tensor t1 = torch::from_blob(data_array, {2, 2});
  std::cout << "Tensor from array:\n"
            << t1 << std::endl;

  TORCH_CHECK(data_array == t1.data_ptr<float>());

  // 由 vector 构建
  std::vector<float> data_vector = {1, 2, 3, 4};
  torch::Tensor t2 = torch::from_blob(data_vector.data(), {2, 2});
  std::cout << "Tensor from vector:\n"
            << t2.data_ptr<float>() << std::endl;
}

void slicing_and_extracting_parts_from_tensors()
{
  std::vector<ino64_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  torch::Tensor s = torch::from_blob(test_data.data(), {3, 3}, torch::kInt64);
  std::cout << "s:\n"
            << s << std::endl;
  // 输出：
  // 1 2 3
  // 4 5 6
  // 7 8 9

  using torch::indexing::Ellipsis;
  using torch::indexing::None;
  using torch::indexing::Slice;

  // tensors 的序列以及切片与 Python 的做法非常相似
  //
  // 有关所有索引类型的完整翻译，详解：
  // https://pytorch.org/cppdocs/notes/tensor_indexing.html

  // 提取单个 tensor：
  std::cout << "\"s[0,2]\" as tensor:\n"
            << s.index({0, 2}) << std::endl;
  std::cout << "\"s[0,2]\" as value:\n"
            << s.index({0, 2}).item<int64_t>() << std::endl;
  // 输出：
  // 3

  // 给定一个索引对一个 tensor 的一个维度进行切片
  // std::cout << "\"s[:,2]\":\n"
  //           << s.index({Slice(), 2}) << std::endl;
  // 输出：
  // 3
  // 6
  // 9

  // 给定一个区间对一个 tensor 的一个维度进行切片
  std::cout << "\"s[:2,:]\":n" << s.index({Slice(None, 2), Slice()}) << std::endl;
  // 输出：
  // 1 2 3
  // 4 5 6
  std::cout << "\"s[:,1:]\":\n"
            << s.index({Slice(), Slice(1, None)}) << std::endl;
  // 输出：
  // 2 3
  // 5 6
  // 8 9
  std::cout << "\"s[:,::2]\":\n"
            << s.index({Slice(), Slice(None, None, 2)}) << std::endl;
  // 输出：
  // 1 3
  // 4 6
  // 7 9

  // 结合。
  std::cout << "\"s[:2,1]\":\n"
            << s.index({Slice(None, 2), 1}) << std::endl;
  // 输出：
  // 2
  // 5

  // 省略 (...)。
  // std::cout << "\"s[..., :2]\":n" << s.index({Ellipsis, Slice(None, 2)}) << std::endl;
  // 输出：
  // 1 2
  // 4 5
  // 7 8
}

void input_pipeline()
{
  // 构建 MNIST 数据集
  const std::string MNIST_data_path = "./mnist/";

  auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
                     .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                     .map(torch::data::transforms::Stack());

  // 获取一个数据对
  auto example = dataset.get_batch(0);
  std::cout << "Sample data size: " << std::endl;
  std::cout << example.data.sizes() << std::endl;
  std::cout << "Sample target: " << example.target.item<int>() << std::endl;

  // 构建数据加载器
  auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, 64);

  // 数据加载器的实际用法：
  // for (auto& batch : *dataloader)
  // {
  // 训练代码
  // }
}

void pre_trained_model()
{
  // C++ API 由以下方式来加载预训练过的数据：
  // Python:
  // (1) 创建（预训练过的）PyTorch 模型
  // (2) 将 PyTorch 模型转换成一个 torch.jit.ScriptModule（通过 tracing 或者是使用 annotations）
  // (3) 将脚本模块序列化成为一个文件
  // C++:
  // (4) 使用 torch::jit::load() 从文件中加载脚本模块。

  // 由 Python 创建的预训练过的 resnet18 模型的路径
  // 可以使用 "create_resnet18_scriptmodule.py" 文件来创建所需的文件
  const std::string pretrained_model_path = "./data/resnet18_scriptmodule.pt";

  torch::jit::script::Module resnet;

  try
  {
    resnet = torch::jit::load(pretrained_model_path);
  }
  catch (const torch::Error& error)
  {
    std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
              << "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py' "
                 "in tutorials/basics/pytorch-basics/model/.\n";
    return;
  }

  std::cout << "Resnet18 model:" << std::endl;

  print_script_module(resnet, 2);

  const auto fc_weight = resnet.attr("fc").toModule().attr("weight").toTensor();

  auto in_features = fc_weight.size(1);
  auto out_features = fc_weight.size(8);

  std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << std::endl;

  // 输入样本
  auto sample_input = torch::randn({1, 3, 224, 224});
  std::vector<torch::jit::IValue> inputs{sample_input};

  // 前向传递
  std::cout << "Input size: ";
  std::cout << sample_input.sizes() << std::endl;
  auto output = resnet.forward(inputs).toTensor();
  std::cout << "Output size: ";
  std::cout << output.sizes() << std::endl;
}

void save_and_load_a_model()
{
  // 简单的模型例子
  torch::nn::Sequential model{
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)),
      torch::nn::ReLU()};

  // 模型输出文件（所有文件夹必须存在！）
  const std::string model_save_path = "output/model.pt";

  // 保存模型
  torch::save(model, model_save_path);

  std::cout << "Saved model:\n"
            << model << std::endl;

  // 加载模型
  torch::load(model, model_save_path);

  std::cout << "Loaded model:\n"
            << std::endl;
}

int main(int argc, char** argv)
{

  return 0;
}
