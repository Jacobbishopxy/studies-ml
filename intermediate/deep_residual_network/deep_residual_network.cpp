/**
 * @file:	deep_residual_network.cpp
 * @author:	Jacob Xie
 * @date:	2023/04/13 21:25:39 Thursday
 * @brief:
 **/

#include "cifar10.h"
#include "resnet.h"
#include "transform.h"
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

using resnet::ResidualBlock;
using resnet::ResNet;
using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main(int argc, char** argv)
{
  std::cout << "Deep Residual Network" << std::endl;

  // 设备
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

  // 超参
  const int64_t num_classes = 10;
  const int64_t batch_size = 100;
  const size_t num_epochs = 20;
  const double learning_rate = 0.001;
  const size_t learning_rate_decay_frequency = 8; // 学习率衰减的 epoch 数
  const double learning_rate_decay_factor = 1.0 / 3.0;

  const std::string cifar10_data_path = std::string{_DATASETS_PATH} + "/cifar10";

  auto train_dataset = CIFAR10(cifar10_data_path)
                           .map(ConstantPad(4))
                           .map(RandomHorizontalFlip())
                           .map(RandomCrop({32, 32}))
                           .map(torch::data::transforms::Stack<>());

  // 训练集样本数
  auto num_train_samples = train_dataset.size().value();

  auto test_dataset = CIFAR10(cifar10_data_path, CIFAR10::Mode::kTest)
                          .map(torch::data::transforms::Stack<>());

  // 测试集样本数
  auto num_test_samples = test_dataset.size().value();

  // 数据加载
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(train_dataset),
      batch_size
  );
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(test_dataset),
      batch_size
  );

  // 模型
  std::array<int64_t, 3> layers{2, 2, 2};
  ResNet<ResidualBlock> model(layers, num_classes);
  model->to(device);

  // 优化器
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

  // 设置输出精度
  std::cout << std::fixed << std::setprecision(4);

  auto current_learning_rate = learning_rate;

  std::cout << "Training..." << std::endl;

  // 训练模型
  for (size_t epoch = 0; epoch != num_epochs; ++epoch)
  {
    // 初始化运行矩阵
    double running_loss = 0;
    size_t num_correct = 0;

    for (auto& batch : *train_loader)
    {
      // 图像与目标标记输入至设备
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      // 前向传递
      auto output = model->forward(data);

      // 计算损失
      auto loss = torch::nn::functional::cross_entropy(output, target);

      // 更新运行损失
      running_loss += loss.item<double>() * data.size(0);

      // 计算预测
      auto prediction = output.argmax(1);

      // 更新被正确分类的样本数
      num_correct += prediction.eq(target).sum().item<int64_t>();

      // 反向传递与优化
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }

    // 衰减学习率
    if ((epoch + 1) % learning_rate_decay_frequency == 0)
    {
      current_learning_rate *= learning_rate_decay_factor;
      static_cast<torch::optim::AdagradOptions&>(
          optimizer
              .param_groups()
              .front()
              .options()
      )
          .lr(current_learning_rate);

      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;

      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
    }

    std::cout << "Training finished!" << std::endl;
    std::cout << "Testing..." << std::endl;

    // 测试模型
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader)
    {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      auto output = model->forward(data);

      auto loss = torch::nn::functional::cross_entropy(output, target);
      running_loss += loss.item<double>() * data.size(0);

      auto prediction = output.argmax(1);
      num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
  }
}
