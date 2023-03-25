/**
 * @file:		convnet.h
 * @author:	Jacob Xie
 * @date:		2023/03/24 22:04:06 Friday
 * @brief:
 *
 * https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/convolutional_neural_network/include/convnet.h
 **/

#pragma once

#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module
{
public:
  explicit ConvNetImpl(int64_t num_classes = 10);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Sequential layer1{
      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
      torch::nn::BatchNorm2d(16),
      torch::nn::ReLU(),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

  torch::nn::Sequential layer2{
      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
      torch::nn::BatchNorm2d(32),
      torch::nn::ReLU(),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))};

  torch::nn::Sequential layer3{
      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
      torch::nn::BatchNorm2d(64),
      torch::nn::ReLU(),
  };

  torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({4, 4})};

  torch::nn::Linear fc;
};

TORCH_MODULE(ConvNet);
