/**
 * @file:	linear_regression.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/11 11:17:16 Saturday
 * @brief:
 **/

#include <iomanip>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char** argv)
{
  std::cout << "Linear Regression" << std::endl;

  // 设备
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

  // 超参
  const int64_t input_size = 1;
  const int64_t output_size = 1;
  const size_t num_epochs = 60;
  const double learning_rate = 0.001;

  // 样本数据库
  auto x_train = torch::randint(0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));
  auto y_train = torch::randint(0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));

  // 线性回归模型
  torch::nn::Linear model(input_size, output_size);
  model->to(device);

  // 优化器
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

  // 设置浮点精度
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "Training..." << std::endl;

  // 训练模型
  for (size_t epoch = 0; epoch != num_epochs; ++epoch)
  {
    // 前向传递
    auto output = model->forward(x_train);
    auto loss = torch::nn::functional::mse_loss(output, y_train);

    // 反向传递及优化
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 5 == 0)
    {
      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], loss: " << loss.item<double>() << std::endl;
    }
  }

  std::cout << "Training finished!" << std::endl;

  return 0;
}
