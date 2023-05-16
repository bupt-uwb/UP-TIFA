import scipy.io as sio
import numpy as np
import pylab as plt
import math


def wgn(x, snr):  # 输出为高斯白噪声
   '''
   程序中用hist()检查噪声是否是高斯分布，psd()检查功率谱密度是否为常数。
   '''
   snr = 10**(snr/10.0)
   xpower = np.sum(x**2)/len(x)
   npower = xpower / snr
   return np.random.randn(len(x)) * np.sqrt(npower)


dataset = sio.loadmat('./data/Amplitude_Data_AGC_BN_Hainan.mat')
X = dataset['data']
Y = dataset['label']
X1 = np.flip(X, axis=1)
X2 = np.vstack([X, X1])
X_Aug = np.zeros([X2.shape[0],X2.shape[1]])
Y_Aug = np.vstack([Y, Y])
# print(Y_Aug.shape)

for i in range(X2.shape[0]):
    n = wgn(X2[i,:], 6)
    X_Aug[i,:] = X2[i,:] + n
# print(X_Aug.shape)

# n = wgn(X2[3852,:], 6)
# print(10*math.log10(sum(X2[3852,:]**2) / sum(n**2)))  # 验算信噪比

sio.savemat('./data/Amplitude_Data_AGC_BN_Hainan_Aug.mat', {'data': X_Aug, 'label': Y_Aug})