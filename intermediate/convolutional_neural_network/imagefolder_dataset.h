/**
 * @file:		imagefolder_dataset.h
 * @author:	Jacob Xie
 * @date:		2023/03/24 22:31:32 Friday
 * @brief:
 **/

#pragma once

#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

namespace dataset
{
/**
 * Dataset 类用于提供 image-label 样本
 */
class ImageFolderDataset : public torch::data::datasets::Dataset<ImageFolderDataset>
{
public:
  enum class Mode
  {
    TRAIN,
    VAL
  };

  explicit ImageFolderDataset(
      const std::string& root,
      Mode mode = Mode::TRAIN,
      torch::IntArrayRef image_load_size = {}
  );

  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override;

private:
  Mode mode_;
  std::vector<int64_t> image_load_size_;
  std::string mode_dir_;
  std::vector<std::string> classes_;
  std::unordered_map<std::string, int> class_to_index_;
  std::vector<std::pair<std::string, int>> samples_;
};

} // namespace dataset
