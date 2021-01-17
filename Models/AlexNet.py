import numpy as np
import torch.nn as nn
import torch

print("Loading data...")
# Load image data (227*227*3)
training_data = np.load("../TrainingData/training_garbage_data_AlexNet.npy", allow_pickle=True)
print("Data is loaded.")
# training_data: 2527张图
# 每张图有2个维度：图片和标签
# 图像：227*227*3

# 参考借鉴：
# https://blog.csdn.net/Gilgame/article/details/85056344?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 227*227*3
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)  # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(   #input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)    #output_size = 6*6*256
        )

        # Fully-connected layer
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 6)
        )

    # 正向传播过程
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

