import torch.nn as nn
import torch
from torch.nn.utils import weight_norm

class CNN(nn.Module):
    def __init__(self,in_ch,mid_ch,out_ch,kernel_size):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, kernel_size=kernel_size,padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(mid_ch),
            nn.Conv1d(mid_ch, mid_ch, kernel_size=kernel_size,padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(mid_ch),
            nn.Conv1d(mid_ch, out_ch, kernel_size=3,padding=(3 - 1) // 2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_ch),
            nn.MaxPool1d(3, 2),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        output = self.conv_block(x)
        # print(output.size())
        return output


class LSTM(nn.Module):

    def __init__(self,in_ch,out_ch,bidirectional):
        super().__init__()
        self.LSTM1 = nn.LSTM(input_size=in_ch, hidden_size=out_ch, num_layers=1, bidirectional=bidirectional,
                             batch_first=True)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x, (h_n, c_n) = self.LSTM1(x)
        return x


class FC(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(FC, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_ch, 100),
            # nn.Linear(1000, 100),
            nn.Linear(100, out_ch)
        )

    def forward(self, x):
        output = self.fc_layer(x)
        # print(output.size())
        return output


class Transformer(nn.Module):
    def __init__(self,in_ch):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_ch,nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=6)

    def forward(self, x):
        output = self.transformer_encoder(x)
        # print(output.size())
        return output


class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()


class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        # shortcut connect
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)


class TemporalConvolutionNetwork(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)  # [N,C_out,L_out=L_in]
        # print(y.shape)
        return self.linear(y[:, :, -1])


class MultiHead_Attention(nn.Module):
    def __init__(self, in_ch, num_head):
        '''
        Args:
            dim: dimension for each time step
            num_head:num head for multi-head self-attention
        '''
        super().__init__()
        self.dim = in_ch
        self.num_head = num_head
        self.qkv = nn.Linear(in_ch, in_ch * 3)  # extend the dimension for later spliting

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2)
        att = att.softmax(dim=1)  # 将多个注意力矩阵合并为一个
        x = (att @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        return x
