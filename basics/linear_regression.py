# @file:	linear_regression.py
# @author:	Jacob Xie
# @date:	2023/03/11 11:17:19 Saturday
# @brief:

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = np.random.uniform(low=0, high=10, size=(15, 1))
y_train = np.random.uniform(low=0, high=10, size=(15, 1))

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))


predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, "ro", label="Original data")
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
plt.show()

torch.save(model.state_dict(), "model.ckpt")
