/**
 * @file:		neural_net.h
 * @author:	Jacob Xie
 * @date:		2023/03/12 09:17:28 Sunday
 * @brief:
 **/

#pragma once

#include <torch/torch.h>

class NeuralNetImpl : public torch::nn::Module
{
public:
  NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes);

  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

TORCH_MODULE(NeuralNet);
