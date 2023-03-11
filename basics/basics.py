# @file:	basics.py
# @author:	Jacob Xie
# @date:	2023/03/10 20:53:36 Friday
# @brief:

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data  # 必要的 import，详见 https://stackoverflow.com/a/47485232


def basic_autograde():
    x = torch.tensor(1.0, requires_grad=True)
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    y = w * x + b

    y.backward()

    print(x.grad)
    print(w.grad)
    print(b.grad)


def basic_autograde2():
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)

    linear = nn.Linear(3, 2)
    print("w: ", linear.weight)
    print("b: ", linear.bias)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    pred = linear(x)

    loss = criterion(pred, y)
    print("loss: ", loss.item())

    loss.backward()

    print("dL/dw: ", linear.weight.grad)
    print("dL/db: ", linear.bias.grad)

    optimizer.step()

    pred = linear(x)
    loss = criterion(pred, y)
    print("loss after 1 step optimization: ", loss.item())


def create_tensors_from_existing_data():
    x = np.array([[1, 2], [3, 4]])
    y = torch.from_numpy(x)
    z = y.numpy()

    print(z)


def input_pipeline():
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    image, label = train_dataset[0]
    print(image.size())
    print(label)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True
    )


def pre_trained_model():
    resnet = torchvision.models.resnet18(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Linear(resnet.fc.in_features, 100)

    images = torch.randn(64, 3, 224, 224)
    outputs = resnet(images)
    print(outputs.size())


def save_and_load_the_model(resnet: torchvision.models.RegNet):
    torch.save(resnet, "model.ckpt")
    model = torch.load("model.ckpt")

    torch.save(resnet.state_dict(), "params.ckpt")
    resnet.load_state_dict(torch.load("params.ckpt"))
