#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 数据采集
from __future__ import print_function
import numpy as np
from numpy import *
import time
from pymoduleconnector import create_mc
import threading
import scipy.io as sio
import os
tmp1 = []
tmp1 = np.array(tmp1)


def radar(com,name):
    # count=8

    # print(count)
    with create_mc(com) as mc:
        xep = mc.get_xep()

        # inti x4driver
        xep.x4driver_init()

        # Set enable pin
        xep.x4driver_set_enable(1)

        # Set iterations
        xep.x4driver_set_iterations(64)
        # Set pulses per step
        xep.x4driver_set_pulses_per_step(5)
        # Set dac step
        xep.x4driver_set_dac_step(1)
        # Set dac min
        xep.x4driver_set_dac_min(949)
        # Set dac max
        xep.x4driver_set_dac_max(1100)
        # Set TX power
        xep.x4driver_set_tx_power(3)  #已修改，原为2

        # Enable downconversion
        xep.x4driver_set_downconversion(0)

        # Set frame area offset
        xep.x4driver_set_frame_area_offset(0.18)
        # offset = xep.x4driver_get_frame_area_offset()

        # Set frame area
        xep.x4driver_set_frame_area(0.2, 9.9)
        # frame_area = xep.x4driver_get_frame_area()

        # Set TX center freq
        xep.x4driver_set_tx_center_frequency(3)

        # Set PRFdiv
        xep.x4driver_set_prf_div(16)
        # prf_div = xep.x4driver_get_prf_div()

        # Start streaming
        xep.x4driver_set_fps(50)

        # fps = xep.x4driver_get_fps()

        def read_frame():
            """Gets frame data from module"""
            d = xep.read_message_data_float()
            frame = np.array(d.data)
            # Convert the resulting frame to a complex array if downconversion is enabled
            return frame

        # Stop streaming

        print( " wait")
        save1 = np.ones((50*62, 1509))

        for jj in range(50*62):

            frame2 = read_frame()

            save1[jj, 0:1509] = frame2
            ll2 = time.asctime(time.localtime(time.time()))

            ll21 = ll2[11:13]
            ll22 = ll2[14:16]
            ll23 = ll2[17:19]

            lll = int(ll21) * 10000 + int(ll22) * 100 + int(ll23)
            # save1[jj, -1] = lll

            print("count:{},time:{}".format(jj,str(lll)))

        sio.savemat('E:/Study/UWB/301/Count_people/Neural_Network/data/Data0-10/Data0-10/20230115_'+name+'_0人9.mat',{'data':save1})
        # count+=1
        print('radar!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        xep.module_reset()




if __name__=='__main__':
    com1 = 'COM3'
    com2 = 'COM9'
    name1 = 'radar1'
    name2 = 'radar2'
    files = os.listdir()
    for file in files:
        if not os.path.isdir(file):
            if file[0:5] == 'debug':
                os.remove(file)
    t1=threading.Thread(name='radar',target=radar,args=(com1,name1))
    # t2 = threading.Thread(name='radar', target=radar, args=(com2,name2))
    # t2=threading.Thread(name='video',target=video)

    t1.start()
    #t2.start()



