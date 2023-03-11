/**
 * @file:	example-app.cpp
 * @author:	Jacob Xie
 * @date:	2023/03/08 23:44:18 Wednesday
 * @brief:
 **/

#include <iostream>
#include <torch/torch.h>

int main()
{
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
