/**
 * @file:		cifar10.h
 * @author:	Jacob Xie
 * @date:		2023/04/10 20:39:47 Monday
 * @brief:
 *
 * https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/deep_residual_network/include/cifar10.h
 **/

#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

// CIFAR10 数据集
// 根据：https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h.
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
  enum Mode
  {
    kTrain,
    kTest
  };

  // 由 `root` 路径加载 CIFAR10 数据集
  // CIFAR10 数据集（二进制版本），由 http://www.cs.toronto.edu/~kriz/cifar.html 提供
  explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);

  // 给定 `index` 返回 `Example`
  torch::data::Example<> get(size_t index) override;

  // 返回数据集大小
  torch::optional<size_t> size() const override;

  // 如果是 CIFAR10 子训练集，返回 true
  bool is_train() const noexcept;

  // 返回所有 images，堆叠至单个 tensor
  const torch::Tensor& images() const;

  // 返回所有 targets，堆叠至单个 tensor
  const torch::Tensor& targets() const;

private:
  torch::Tensor images_;
  torch::Tensor targets_;
  Mode mode_;
};
