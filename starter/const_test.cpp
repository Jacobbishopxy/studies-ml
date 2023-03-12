/**
 * @file:		env_test.cpp
 * @author:	Jacob Xie
 * @date:		2023/03/12 12:31:53 Sunday
 * @brief:  CONST test
 **/

#include <iostream>
#include <string>

#ifdef _DATASETS_PATH
const std::string datasets_path = _DATASETS_PATH;
#endif // _DATASETS_PATH

int main(int argc, char** argv)
{

  std::string MNIST_data_path = datasets_path + "/mnist";

  std::cout << "MNIST_data_path: " << MNIST_data_path << std::endl;

  return 0;
}
