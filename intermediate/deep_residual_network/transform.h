/**
 * @file:		transform.h
 * @author:	Jacob Xie
 * @date:		2023/04/10 21:44:20 Monday
 * @brief:
 **/

#pragma once

#include <random>
#include <torch/torch.h>
#include <vector>

namespace transform
{
class RandomHorizontalFlip : public torch::data::transforms::TensorTransform<torch::Tensor>
{
public:
  // 创建一个 transformation 用于随机横向翻转一个 tensor
  //
  // 参数 `p` 决定了反转的概率
  explicit RandomHorizontalFlip(double p = 0.5);

  torch::Tensor operator()(torch::Tensor input) override;

private:
  double p_;
};

class ConstantPad : public torch::data::transforms::TensorTransform<torch::Tensor>
{
public:
  // 创建一个 transformation 用于填充一个 tensor 边缘
  //
  // `padding` 是一个长度为 4 的向量，用于填充边缘（左、右、上、下）。`value` 决定填充的像素
  explicit ConstantPad(const std::vector<int64_t>& padding, torch::Scalar value = 0);

  explicit ConstantPad(int64_t padding, torch::Scalar value = 0);

  torch::Tensor operator()(torch::Tensor input) override;

private:
  std::vector<int64_t> padding_;
  torch::Scalar value_;
};

class RandomCrop : public torch::data::transforms::TensorTransform<torch::Tensor>
{
public:
  // 创建一个 transformation 用于随机修剪一个 tensor
  //
  // 参数 `size` 是一个长度为 2 的向量，其决定嘞输出长度（高、宽）
  explicit RandomCrop(const std::vector<int64_t>& size);
  torch::Tensor operator()(torch::Tensor input) override;

private:
  std::vector<int64_t> size_;
};

}; // namespace transform