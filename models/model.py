import torch.nn as nn
from .blocks import *
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = CNN(in_ch=1, mid_ch=32, out_ch=64, kernel_size=5)
        self.conv_block2 = CNN(in_ch=1, mid_ch=32, out_ch=64, kernel_size=3)
        self.fc = FC(in_ch=128 * ((1400 - 3) // 2 + 1), out_ch=11)
        # self.lstm = LSTM(in_ch=32, out_ch=32, bidirectional=True)
        # self.transformer = Transformer(in_ch=64)
        self.multihead_attention = MultiHead_Attention(in_ch=128, num_head=8)
        # self.tcn = TCN(64, 10, [64] * 5, 3, 0.2)

    def forward(self, x):
        # y = torch.sum(torch.pow(x, 2), dim=(2,), keepdim=True)
        # x = torch.cat((x,y),dim=2)
        # print(x.shape)
        x1 = x[:,0,:].reshape(x.shape[0],1,-1)
        x2 = x[:,1,:].reshape(x.shape[0],1,-1)
        # print(x1.shape)
        # x = self.conv_block1(x).transpose(1, 2)
        x1 = self.conv_block1(x1).transpose(1, 2)
        x2 = self.conv_block1(x2).transpose(1, 2)
        x = torch.cat((x1, x2), dim=2)
        # print(x.shape)
        # x = self.lstm(x).transpose(1, 2)
        # print(x.shape)
        # x = self.transformer(x)
        # x = x.transpose(0,1)
        # x = x.transpose(1,2)
        # print(x.shape)
        x = self.multihead_attention(x)
        # y = torch.sum(torch.pow(x,2),dim=(1,), keepdim=True)
        # print(x.shape)
        # x = torch.cat((x,y),dim=1)
        # print(x.shape)
        # output = self.tcn(x)
        output = self.fc(x)
        # print(output.shape)
        return output
