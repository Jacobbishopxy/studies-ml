/**
 * @file:	cifar10.cpp
 * @author:	Jacob Xie
 * @date:	2023/04/13 19:17:48 Thursday
 * @brief:
 **/

#include "cifar10.h"

namespace
{
// CIFAR10 数据集的描述在此 https://www.cs.toronto.edu/~kriz/cifar.html.
constexpr uint32_t kTrainsSize = 50000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kSizePerBatch = 10000;
constexpr uint32_t kImageRows = 32;
constexpr uint32_t kImageColumns = 32;
constexpr uint32_t kBytesPerRow = 3073;
constexpr uint32_t kBytesPerChannelPerRow = 1024;
constexpr uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;

const std::vector<std::string> kTrainDataBatchFiles = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
};

const std::vector<std::string> kTestDataBatchFiles = {
    "test_batch.bin",
};

// 源：https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
std::string join_paths(std::string head, const std::string& tail)
{
  if (head.back() != '/')
  {
    head.push_back('/');
  }
  head += tail;
  return head;
}
// 部分根据：https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train)
{
  const auto& files = train ? kTrainDataBatchFiles : kTestDataBatchFiles;
  const auto num_samples = train ? kTrainsSize : kTestSize;

  std::vector<char> data_buffer;
  data_buffer.reserve(files.size() * kBytesPerBatchFile);

  for (const auto& file : files)
  {
    const auto path = join_paths(root, file);
    std::ifstream data(path, std::ios::binary);
    TORCH_CHECK(data, "Error opening data file at", path);

    data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
  }

  TORCH_CHECK(data_buffer.size() == files.size() * kBytesPerBatchFile, "Unexpected file sizes");

  auto targets = torch::empty(num_samples, torch::kByte);
  auto images = torch::empty({num_samples, 3, kImageRows, kImageColumns}, torch::kByte);

  for (uint32_t i = 0; i != num_samples; ++i)
  {
    // 每个 row 的首个 byte 是 target 类的 index
    uint32_t start_index = i * kBytesPerRow;
    targets[i] = data_buffer[start_index];

    // 接下来的 bytes 关系到 rgb channel 值，其顺序如下：
    // red (32 * 32 = 1024 bytes) | green (1024 bytes) | blue (1024 bytes)
    uint32_t image_start = start_index + 1;
    uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
    std::copy(
        data_buffer.begin() + image_start, data_buffer.begin() + image_end,
        reinterpret_cast<char*>(images[i].data_ptr())
    );
  }

  return {images.to(torch::kFloat32).div_(255), targets.to(torch::kI64)};
}
} // namespace

CIFAR10::CIFAR10(const std::string& root, Mode mode) : mode_(mode)
{
  auto data = read_data(root, mode == Mode::kTrain);

  images_ = std::move(data.first);
  targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR10::get(size_t index)
{
  return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR10::size() const
{
  return images_.size(0);
}

bool CIFAR10::is_train() const noexcept
{
  return mode_ == Mode::kTrain;
}

const torch::Tensor& CIFAR10::images() const
{
  return images_;
}

const torch::Tensor& CIFAR10::targets() const
{
  return targets_;
}
