/**
 * @file:		image_io.h
 * @author:	Jacob Xie
 * @date:		2023/03/25 08:36:40 Saturday
 * @brief:
 *
 * https://github.com/prabhuomkar/pytorch-cpp/blob/master/utils/image_io/include/image_io.h
 **/

#pragma once

#include <string>
#include <torch/torch.h>

namespace image_io
{

enum class ImageFormat
{
  PNG,
  JPG,
  BMP
};

torch::Tensor load_image(
    const std::string& file_path,
    torch::IntArrayRef shape = {},
    std::function<torch::Tensor(torch::Tensor)> transform = [](torch::Tensor x)
    { return x; }
);

void save_image(
    torch::Tensor tensor,
    const std::string& file_path,
    int64_t nrow = 10,
    int64_t padding = 2,
    bool normalize = false,
    const std::vector<double>& range = {},
    bool scale_each = false,
    torch::Scalar pad_value = 0,
    ImageFormat format = ImageFormat::PNG
);

} // namespace image_io
