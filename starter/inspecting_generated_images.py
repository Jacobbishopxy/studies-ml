# @file:	inspecting_generated_images.py
# @author:	Jacob Xie
# @date:	2023/03/13 21:15:46 Monday
# @brief:	Inspecting Generated Images

from __future__ import print_function
from __future__ import unicode_literals

import argparse

import matplotlib.pyplot as plt
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--sample-file", required=True)
parser.add_argument("-o", "--out-file", default="out.png")
parser.add_argument("-d", "--dimension", type=int, default=3)
options = parser.parse_args()

# 与原文的 `torch.jit.load` 不同，详见：https://github.com/ultralytics/yolov5/issues/1248#issuecomment-719654953
module = torch.load(options.sample_file)
images = list(module.parameters())[0]

for index in range(options.dimension * options.dimension):
    image = images[index].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
    array = image.numpy()
    axis = plt.subplot(options.dimension, options.dimension, 1 + index)
    plt.imshow(array, cmap="gray")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

plt.savefig(options.out_file)
print("Saved ", options.out_file)
