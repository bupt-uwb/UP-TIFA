#-*-coding:utf-8 -*-
from numpy import *

'''
对雷达相位信号进行预处理，仅去除背景杂波，不过带通, 不去直流。
'''
def p_f_phase(raw_phase_data):
    c = raw_phase_data.shape
    row = c[0]  # 一共多少行
    col = c[1]  # 一共多少列
    # DC_removed = zeros((row, col-2))
    # raw_phase_data = raw_phase_data[:, 35:] ############  ???????疑问
    # 直接把前35列的直达波和最后两列去掉在去杂波
    ClutterData = raw_phase_data.copy()
    pure_phase_data = raw_phase_data.copy()  ########### ??????疑问


    alpha = 0.9
    for i in range(row):
        # selected = raw_phase_data[i,:-3]
        # for j in range(col - 2):
        #     DC_removed[i, j] = raw_phase_data[i, j] - mean(selected)
        # # 到这里已经逐行地去除了该行的均值
        if i == 0:
            ClutterData[i, :] = (1 - alpha) * raw_phase_data[i, :]
            pure_phase_data[i, :] = raw_phase_data[i, :] - ClutterData[i, :]
        else:
            ClutterData[i, :] = alpha * ClutterData[i - 1, :] + (1 - alpha) * raw_phase_data[i, :]
            pure_phase_data[i, :] = raw_phase_data[i, :] - ClutterData[i, :]
    ############ return raw_phase_data   ???疑问
    return pure_phase_data[50:row,:]

'''
对雷达数据进行预处理带通滤波，只保留有用的频段的数据以及去除背景噪声
Input:
    RawData: 原始雷达信号
    M: 输出前去除信号矩阵的前M行
    L: 输出前去除信号矩阵的前L列
Output:
    PureData: 预处理后的雷达信号
'''
def p_f(RawData,M,L):
    c=RawData.shape  # 数组维度
    FrameStitchnum = int(c[1]/156)  # 距离(单位米)向下取整，后面会以一米为组去除平均值，本脉冲超宽带雷达1m有156个采样点，即156列
    #########带通滤波器，通带为6GHz-8.5GHz，即超宽带雷达的高频段##################
    BanFilter = array([0.00951405530408726,-0.0213256530382494,0.0971269581777537,0.116043772514853,-0.0719213416035251,0.0310751542226853,-0.0117317374641936,0.0155248397824553,-0.00897930830281151,-0.0175257324703966,0.0138210975220308,0.0321622343763948,-0.0518153341702605,0.00862037644819017,0.0319063027179502,-0.0165142275010029,-0.00116361694656361,-0.0261090586203386,0.0259082609081522,0.0611906198141473,-0.118995359095095,0.00698411530218591,0.167853386324139,-0.148088404382625,-0.0815233633716659,0.222983033920004,-0.0815233633716659,-0.148088404382625,0.167853386324139,0.00698411530218591,-0.118995359095095,0.0611906198141473,0.0259082609081522,-0.0261090586203386,-0.00116361694656361,-0.0165142275010029,0.0319063027179502,0.00862037644819017,-0.0518153341702605,0.0321622343763948,0.0138210975220308,-0.0175257324703966,-0.00897930830281151,0.0155248397824553,-0.0117317374641936,0.0310751542226853,-0.0719213416035251,0.116043772514853,0.0971269581777537,-0.0213256530382494,0.00951405530408726])


    #########数据与预设##################
    batches=c[0]  # 一共有多少行数据
    BandpassData = zeros((c[0],c[1]))  # 经过带通滤波器后的雷达数据
    ClutterData = zeros((c[0],c[1]))  # 静态杂波数据
    PureData = zeros((c[0],c[1]))  # 滤波后的雷达数据
    pnum=156  # 1m有156个采样点
    firnum=50  # 卷积边界
    alpha=0.9  # 修正因子
    #############################预处理######################################
    for row in range(batches):  # 对每行数据分别进行预处理
        # 每行、每米（每156列）的数据去除该段数据平均值，最后列数不满的1米的部分也视为一组处理
        for framenum in range(FrameStitchnum):
            blockdata=RawData[row, (framenum) * pnum:min((framenum + 1) * pnum, c[1])]
            blockmean=mean(blockdata)  # 平均值

            DCmean=ones((1,blockdata.shape[0])) * blockmean  # 平均值转换为等长向量
            RawData[row, (framenum) * pnum:min((framenum + 1) * pnum, c[1])] = blockdata - DCmean

        # print('raw')
        # print(RawData[row,:])
        convres=convolve(RawData[row,:], BanFilter)  # 一行原始信号与带通滤波器时域卷积，频域相乘
        # print('pass')
        # print(convres)

        BandpassData[row,:] = convres[int(firnum/2):int(firnum/2 + c[1])]
        if row==0:  # 如果是第一行
            ClutterData[row, :]=(1 - alpha)*BandpassData[row, :]  # 第一行静态噪声数据初始化
            PureData[row, :]=BandpassData[row, :] - ClutterData[row, :]  # 信号减去静态噪声数据
        if row>0:  # 其他
            # 后续静态噪声为前一行静态噪声与本行带通滤波后数据的加权和，权重为alpha=0.9
            ClutterData[row, :] = alpha * ClutterData[row-1, :] + (1 - alpha) * BandpassData[row, :] 
            PureData[row, :] = BandpassData[row, :] - ClutterData[row, :]

    PureData=PureData[M:c[0],L:]  # 去除前20(M)行，去除前50(L)列(直达波信号)

    return PureData

'''
SVD预处理
'''

def radarnorm(radar):
    radar = delete(radar, -1, axis=1)
    rsize = radar.shape
    temp = zeros(rsize)
    for i in range(rsize[0]):
        for j in range(rsize[1]):
            rmax = max(radar[i, :])
            rmin = min(radar[i, :])
            temp[i, j] = (radar[i, j] - rmin) / (rmax - rmin)
    normed_radar = temp
    return normed_radar

def clutter_svd(radar):
    U, S, Vt = linalg.svd(radar)
    sigma1 = S[0]
    u1 = U[:, 0]
    u1 = u1.reshape(-1, 1)
    vt1 = Vt[0, :]
    vt1 = vt1.reshape(1, -1)
    noise = sigma1 * u1 * vt1
    radar_nc = radar - noise
    return radar_nc

def prepocess(radar):
    radar_n = radarnorm(radar)
    radar_nc = clutter_svd(radar_n)
    return radar_nc







