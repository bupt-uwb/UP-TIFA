import json
import numpy as np
import os
import scipy.io as scio
import pca_filter
from sklearn import preprocessing, decomposition
import scipy.io as sio
import matplotlib.pyplot as plt


def bartlett_periodogram(x, nsegments, nfft):
    nperseg = len(x) // nsegments
    psd = np.zeros(nfft)
    for i in range(nsegments):
        segment = x[i * nperseg:(i + 1) * nperseg]
        psd += np.abs(np.fft.fft(segment, nfft)) ** 2 / nfft
    psd[0] = 0
    psd /= nsegments
    # psd = psd[0: nfft // 2]
    return psd


def string20(txt,width=20):
    """
    对字符串内的数字前添加n个'0'，n = width - 数字长度， 如：-1-2-  ->  -00000000000000000001-00000000000000000002-
    该函数主要应用于更正 os模块遍历文件时的排序方式。
    :param txt: 原始字符串
    :param width: 处理后的数字长度
    :return: 处理后的字符串
    """
    result = ''
    digital = ''
    for s in txt:
        if s.isdigit():
            digital += s
        else:
            if len(digital) == 0:
                result += s
            else:
                result += '0'*(width-len(digital)) + digital + s
                digital = ''
    if len(digital)!=0:  # 当字符串末尾是数字时，循环结束后要单独再处理
        result += '0' * (width - len(digital)) + digital + s
    return result


fileDir = './data/Data0-10/Data0-10/'
# fileDir = './data/海南测试/丛林/'
DATA = []

Np = 100
Pure_Data, Pure_Data_AGC = [], []
Amplitude_Data, Amplitude_Data_AGC = [], []
Fre = []
Psd = []

for i in sorted(os.listdir(fileDir),key=string20):
    if os.path.splitext(i)[1] == '.mat':
        DATA.append(i)
print(DATA)

# label = scio.loadmat('./data/Data0-10/label.mat')
# truth = label['label']

for j in range(len(DATA)):
    print(str(j)+'/'+str(len(DATA)))
    data = scio.loadmat(fileDir + DATA[j])
    RawData = data['data']
    pure_data = pca_filter.p_f(RawData, Np // 2, 50)
    # pure_data = RawData[50:,50:]
    fc = 7.29e9
    fs = 23.328e9
    low_filter = [0.000389120071389383, 0.0949761173581931, 0.00580329087790910, 0.00472425053890488,
                  0.00239884306254489, -0.000439302446221061, -0.00333038134587463, -0.00575506510785088,
                  -0.00728926014588025, -0.00760598322947964, -0.00660723560639280, -0.00439064328852855,
                  -0.00129628413448493, 0.00220340419217754, 0.00549950113357075, 0.00802235033766739,
                  0.00926300566051567, 0.00895731942493143, 0.00701842266592757, 0.00372529405181510,
                  -0.000372039527776018, -0.00487704483110525, -0.00869624580409710, -0.0112566686034905,
                  -0.0120175241847411, -0.0107287771842090, -0.00739206054035988, -0.00244587526203289,
                  0.00341754785694278, 0.00922008657059010, 0.0139500458424295, 0.0166397021318728, 0.0166199012938838,
                  0.0135602277340880, 0.00764885754766772, -0.000467062065668899, -0.00964552748312256,
                  -0.0184726725502639, -0.0253182741049292, -0.0286657543584366, -0.0273092414175282,
                  -0.0204043636510659, -0.00781539726932600, 0.00988882872141378, 0.0315171291518322,
                  0.0553266299201289, 0.0791577153232821, 0.100754217391452, 0.117965642991575, 0.129056358673261,
                  0.132880896170082, 0.129056358673261, 0.117965642991575, 0.100754217391452, 0.0791577153232821,
                  0.0553266299201289, 0.0315171291518322, 0.00988882872141378, -0.00781539726932600,
                  -0.0204043636510659, -0.0273092414175282, -0.0286657543584366, -0.0253182741049292,
                  -0.0184726725502639, -0.00964552748312256, -0.000467062065668899, 0.00764885754766772,
                  0.0135602277340880, 0.0166199012938838, 0.0166397021318728, 0.0139500458424295, 0.00922008657059010,
                  0.00341754785694278, -0.00244587526203289, -0.00739206054035988, -0.0107287771842090,
                  -0.0120175241847411, -0.0112566686034905, -0.00869624580409710, -0.00487704483110525,
                  -0.000372039527776018, 0.00372529405181510, 0.00701842266592757, 0.00895731942493143,
                  0.00926300566051567, 0.00802235033766739, 0.00549950113357075, 0.00220340419217754,
                  -0.00129628413448493, -0.00439064328852855, -0.00660723560639280, -0.00760598322947964,
                  -0.00728926014588025, -0.00575506510785088, -0.00333038134587463, -0.000439302446221061,
                  0.00239884306254489, 0.00472425053890488, 0.00580329087790910, 0.0949761173581931,
                  0.000389120071389383]
    Amplitude_data = []
    for p in range(pure_data.shape[0]):
        frame = pure_data[p][:]
        csine = np.exp(-1j * fc / fs * 2 * np.pi * np.arange(frame.shape[0]))
        cframe = frame * csine
        cframe_lp = np.convolve(cframe, low_filter)
        Amplitude = np.abs(cframe_lp[50:-50])
        Amplitude_data.append(Amplitude)
    Amplitude_data = np.array(Amplitude_data)

    for s in range(0, pure_data.shape[0] - Np // 2, Np // 2):  # 2秒窗口长度，每1秒滑动切片
        Data = pure_data[s:s + Np, :].copy()
        # Data2 = Amplitude_data[s:s + Np, :].copy()

        signal1 = Data.copy()
        # signal2 = Data2.copy()
        #
        signal = np.zeros([Np, Data.shape[1]])
        low_filter3Hz = [-0.000117180958199210, -0.0960191602656036, -0.0293178021939186, -0.0311647878099892,
                         -0.0310676396656854, -0.0289824002275070, -0.0250663500370471, -0.0196072606454893,
                         -0.0131366173332873, -0.00623229404028813, 0.000387862073289421, 0.00610554514392253,
                         0.0103381759950058, 0.0127123272147993, 0.0130361729235399, 0.0114047325242074,
                         0.00807710143043115,
                         0.00360101568064756, -0.00141143722818117, -0.00619845854743972, -0.0101079803797637,
                         -0.0125303430764751, -0.0130679749055136, -0.0117033531827936, -0.00834286186811615,
                         -0.00362237026033357, 0.00188620205764982, 0.00751115934564799, 0.0123561955634699,
                         0.0156325304310312,
                         0.0167257140595351, 0.0153602835220221, 0.0114826833379818, 0.00543252907385482,
                         -0.00215369582126381,
                         -0.0103353211483626, -0.0179849368050812, -0.0238814246033293, -0.0269356127524514,
                         -0.0262015282897873, -0.0211324889902496, -0.0115275032695307, 0.00228944016745521,
                         0.0196117994494655,
                         0.0392897018516023, 0.0598650620754721, 0.0797475003250911, 0.0972811656794309,
                         0.110981054315352,
                         0.119733830695567, 0.122735325145847, 0.119733830695567, 0.110981054315352, 0.0972811656794309,
                         0.0797475003250911, 0.0598650620754721, 0.0392897018516023, 0.0196117994494655,
                         0.00228944016745521,
                         -0.0115275032695307, -0.0211324889902496, -0.0262015282897873, -0.0269356127524514,
                         -0.0238814246033293, -0.0179849368050812, -0.0103353211483626, -0.00215369582126381,
                         0.00543252907385482, 0.0114826833379818, 0.0153602835220221, 0.0167257140595351,
                         0.0156325304310312,
                         0.0123561955634699, 0.00751115934564799, 0.00188620205764982, -0.00362237026033357,
                         -0.00834286186811615, -0.0117033531827936, -0.0130679749055136, -0.0125303430764751,
                         -0.0101079803797637, -0.00619845854743972, -0.00141143722818117, 0.00360101568064756,
                         0.00807710143043115, 0.0114047325242074, 0.0130361729235399, 0.0127123272147993,
                         0.0103381759950058,
                         0.00610554514392253, 0.000387862073289421, -0.00623229404028813, -0.0131366173332873,
                         -0.0196072606454893, -0.0250663500370471, -0.0289824002275070, -0.0310676396656854,
                         -0.0311647878099892, -0.0293178021939186, -0.0960191602656036,
                         -0.000117180958199210]  # 3Hz低通滤波器，FIR
        for n in range(signal1.shape[1]):
            temp1 = np.convolve(signal1[:, n], low_filter3Hz)
            temp2 = temp1[50:-50]
            signal[:, n] = temp2
        fre = abs(np.fft.fft(signal, axis=0))

        for i in range(signal1.shape[0]):  # AGC(若添加，则阈值也调整增加，求噪声标准差不添加AGC)
            for ii in range(signal1.shape[1]):
                signal1[i, ii] = (ii / 156 + 0.5) ** 1.5 * signal1[i, ii]

        # for i in range(signal2.shape[0]):  # AGC(若添加，则阈值也调整增加，求噪声标准差不添加AGC)
        #     for ii in range(signal2.shape[1]):
        #         signal2[i, ii] = (ii / 156 + 0.5) ** 1.5 * signal2[i, ii]
        #
        for i in range(fre.shape[0]):  # AGC(若添加，则阈值也调整增加，求噪声标准差不添加AGC)
            for ii in range(fre.shape[1]):
                fre[i, ii] = (ii / 156 + 0.5) ** 1.5 * fre[i, ii]

        # PSD = np.zeros([Data.shape[0],Data.shape[1]])
        # for k in range(Data.shape[0]):
        #     psd = bartlett_periodogram(Data[k, :], 10, Data.shape[1])
        #     PSD[k,:] = psd
            # print(PSD.shape)

        # Data = np.mean(np.abs(Data), axis=0)
        # Data2 = np.mean(np.abs(Data2), axis=0)
        signal1 = np.mean(np.abs(signal1), axis=0)
        # signal2 = np.mean(np.abs(signal2), axis=0)
        fre = np.mean(np.abs(fre), axis=0)
        # PSD = np.mean(np.abs(PSD), axis=0)

        # Data -= Data.mean(axis=0)
        # Data /= Data.std(axis=0)
        # Data2 -= Data2.mean(axis=0)
        # Data2 /= Data2.std(axis=0)
        signal1 -= signal1.mean(axis=0)
        signal1 /= signal1.std(axis=0)
        # signal2 -= signal2.mean(axis=0)
        # signal2 /= signal2.std(axis=0)
        fre -= fre.mean(axis=0)
        fre /= fre.std(axis=0)
        # PSD -= PSD.mean(axis=0)
        # PSD /= PSD.std(axis=0)

        # print(PSD.shape,signal2.shape)

        # Pure_Data.append(Data)
        # Amplitude_Data.append(Data2)
        Pure_Data_AGC.append(signal1)
        # Amplitude_Data_AGC.append(signal2)
        Fre.append(fre)
        # Psd.append(PSD)

# Amplitude_Data_AGC_Fre = np.stack((Amplitude_Data_AGC,Fre)).transpose((1,0,2))
Pure_Data_AGC_Fre = np.stack((Pure_Data_AGC,Fre)).transpose((1,0,2))
# Amplitude_Data_AGC = np.stack((Amplitude_Data_AGC,Psd)).transpose((1,0,2))

sio.savemat('./data/Data0-10/Raw_Data_AGC_BN.mat', {'data': np.array(Pure_Data_AGC)})
# sio.savemat('./data/Data0-10/Amplitude_Data_AGC_BN.mat', {'data': np.array(Amplitude_Data_AGC)})
sio.savemat('./data/Data0-10/Raw_Data_AGC_BN_Fre.mat', {'data': np.array(Pure_Data_AGC_Fre)})
# sio.savemat('./data/Data0-10/Amplitude_Data_AGC_BN_Fre.mat', {'data': np.array(Amplitude_Data_AGC_Fre)})
