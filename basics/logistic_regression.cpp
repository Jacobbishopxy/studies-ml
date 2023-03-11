/**
 * @file:	logistic_regression.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/11 12:52:19 Saturday
 * @brief:
 *
 * https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/basics/logistic_regression/main.cpp
 **/

#include <iomanip>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char** argv)
{
  std::cout << "Logistic Regression" << std::endl;

  // ================================================================================================
  // Param
  // ================================================================================================

  // 设备
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

  // 超参
  const int64_t input_size = 784; // 28 * 28
  const int64_t num_classes = 10;
  const int64_t batch_size = 100;
  const size_t num_epochs = 5;
  const double learning_rate = 0.001;

  // ================================================================================================
  // Data
  // ================================================================================================

  const std::string MNIST_data_path = "./data/mnist";

  // MNIST 数据集（图片与标签）
  auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());

  // 训练集的样本数量
  auto num_train_samples = train_dataset.size().value();

  auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());

  // 测试集的样本数量
  auto num_test_samples = test_dataset.size().value();

  // 数据加载
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(train_dataset), batch_size
  );
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(test_dataset), batch_size
  );

  // ================================================================================================
  // Model
  // ================================================================================================

  // 对数回归模型
  torch::nn::Linear model(input_size, num_classes);

  model->to(device);

  // 损失与优化器
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

  std::cout << std::fixed << std::setprecision(4);

  // ================================================================================================
  // Train
  // ================================================================================================

  std::cout << "Training..." << std::endl;

  // 训练模型
  for (size_t epoch = 0; epoch != num_epochs; ++epoch)
  {
    double running_loss = 0.0;
    size_t num_correct = 0;

    for (auto& batch : *train_loader)
    {
      auto data = batch.data.view({batch_size, -1}).to(device);
      auto target = batch.target.to(device);

      // 前向传递
      auto output = model->forward(data);

      // 计算损失
      auto loss = torch::nn::functional::cross_entropy(output, target);

      // 更新运行损失
      running_loss += loss.item<double>() * data.size(0);

      // 计算预测
      auto prediction = output.argmax(1);

      // 更新正确的分类数
      num_correct += prediction.eq(target).sum().item<int64_t>();

      // 反向传递与优化
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }

    auto sample_mean_loss = running_loss / num_train_samples;
    auto accuracy = static_cast<double>(num_correct) / num_train_samples;

    std::cout << "Epoch [" << (epoch + 1)
              << "/" << num_epochs
              << "], Trainset - Loss: " << sample_mean_loss
              << ", Accuracy: " << accuracy << std::endl;
  }

  std::cout << "Training finished!" << std::endl;

  // ================================================================================================
  // Test
  // ================================================================================================

  std::cout << "Testing..." << std::endl;

  // 测试模型
  model->eval();
  torch::NoGradGuard no_grad;

  double running_loss = 0.0;
  size_t num_correct = 0;

  for (const auto& batch : *test_loader)
  {
    auto data = batch.data.view({batch_size, -1}).to(device);
    auto target = batch.target.to(device);

    auto output = model->forward(data);

    auto loss = torch ::nn::functional::cross_entropy(output, target);

    running_loss += loss.item<double>() * data.size(0);

    auto prediction = output.argmax(1);

    num_correct += prediction.eq(target).sum().item<int64_t>();
  }

  std::cout << "Testing finished!" << std::endl;

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss / num_test_samples;

  std::cout << "Testset - Loss: " << test_sample_mean_loss
            << ", Accuracy: " << test_accuracy << std::endl;

  return 0;
}
