# @file:	define_nn_models.py
# @author:	Jacob Xie
# @date:	2023/03/07 16:32:45 Tuesday
# @brief:

import torch


class Net1(torch.nn.Module):
    def __init__(self, N, M) -> None:
        super(Net1, self).__init__()
        self.W = torch.nn.parameter.Parameter(torch.randn(N, M))
        self.b = torch.nn.parameter.Parameter(torch.randn(M))

    def forward(self, input):
        return torch.addmm(self.b, input, self.W)


class Net2(torch.nn.Module):
    def __init__(self, N, M) -> None:
        super(Net2, self).__init__()
        self.linear = torch.nn.Linear(N, M)
        self.another_bias = torch.nn.parameter.Parameter(torch.rand(M))

    def forward(self, input):
        return self.linear(input) + self.another_bias
