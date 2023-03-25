/**
 * @file:		convolutional_neural_network.cpp
 * @author:	Jacob Xie
 * @date:		2023/03/24 21:53:45 Friday
 * @brief:
 *
 * https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/src/main.cpp
 **/

#include "convnet.h"
#include "imagefolder_dataset.h"
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

using dataset::ImageFolderDataset;

int main(int argc, char** argv)
{
  std::cout << "Convolutional Neural Network" << std::endl;

  // 设备
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CUP") << std::endl;

  // 超参
  const int64_t num_classes = 10;
  const int64_t batch_size = 8;
  const size_t num_epochs = 10;
  const double learning_rate = 1e-3;
  const double weight_decay = 1e-3;

  const std::string imagenette_data_path = std::string{_DATASETS_PATH} + "/imagenette2-160";

  // Imagenette dataset
  // 训练数据集
  auto train_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::TRAIN, {160, 160})
                           .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                           .map(torch::data::transforms::Stack<>());
  // 训练集样本数
  auto num_train_samples = train_dataset.size().value();

  // 测试数据集
  auto test_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::VAL, {160, 160})
                          .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                          .map(torch::data::transforms::Stack<>());
  // 测试集样本数
  auto num_test_samples = test_dataset.size().value();

  // Data loader
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(train_dataset), batch_size
  );
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(test_dataset), batch_size
  );

  // 模型
  ConvNet model(num_classes);
  model->to(device);

  // 优化器
  torch::optim::Adam optimizer(
      model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay)
  );

  // 设置浮点输出精度
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "Training..." << std::endl;

  // 训练模型
  for (size_t epoch = 0; epoch != num_epochs; ++epoch)
  {
    // 初始化运行矩阵
    double running_loss = 0.0;
    size_t num_correct = 0;

    for (auto& batch : *train_loader)
    {
      // 转换图像与目标表情至设备
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      // 前向传递
      auto output = model->forward(data);

      // 计算损失
      auto loss = torch::nn::functional::cross_entropy(output, target);

      // 更新运行损失
      running_loss += loss.item<double>() * data.size(0);

      // 计算精度
      auto prediction = output.argmax(1);

      // 更新正确的样本分类数
      num_correct += prediction.eq(target).sum().item<int64_t>();

      // 反向传递与优化
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }

    // 样本平均损失与精度
    auto sample_mean_loss = running_loss / num_train_samples;
    auto accuracy = static_cast<double>(num_correct) / num_train_samples;

    std::cout << "Epoch [" << (epoch + 1)
              << "/" << num_epochs
              << "], Trainset - Loss:" << sample_mean_loss
              << ", Accuracy: " << accuracy << std::endl;
  }

  std::cout << "Training finished!" << std::endl;
  std::cout << "Testing..." << std::endl;

  // 测试模型
  model->eval();
  torch::InferenceMode no_grad;

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

  std::cout << "Testing finished!" << std::endl;

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss / num_test_samples;

  std::cout << "Testset - Loss:" << test_sample_mean_loss << ", Accuracy: " << test_accuracy << std::endl;

  return 0;
}
