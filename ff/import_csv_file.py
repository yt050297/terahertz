import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from lib.Allread_scanner import allread_scanner

#使用する画像のディレクトリ
dir_path = '/Users/kawaselab/PycharmProjects/scanner/20200702/angle_15do/after_bond'
#画像の枚数
img_num = 10

#分割測定用カットするピクセル数
cut_pixel = 80
#分割測定用保存するディレクトリ
save_path = '/Users/kawaselab/PycharmProjects/scanner/20200702

'''
＃強度分布をグラフ化
for i in range(0,img_num):
    #i = i * 5
    title = 'cardboard_{}'.format(i)
    file_name = '{}_raw.csv'.format(i)
    file = os.path.join(dir_path,file_name)
    #col = ['{0}'.format(j) for j in range(512)]
    try:
        pw_max = allread_scanner(file,title).lightsource_beamshape_smoothing()
    except FileNotFoundError as e:
        print(e)
#allread_scanner(file,title).plot_attenuation(pw_max)


#強度分布をイメージングで可視化
title = 'scanner_intensity_graph'
file_name = 'knife_noalumi_no_norm_raw.csv'
file = os.path.join(dir_path,file_name)
image = allread_scanner(file,title).visuallization()
'''

#分割測定からの画像の再構成
for i in range(0,img_num):
    file_name1 = '{}_raw.csv'.format(i)
    file_name2 = '{}_raw.csv'.format(i)

    file1 = os.path.join(dir_path,file_name1)
    file2 = os.path.join(dir_path, file_name2)

    img1 = cv2.imread(file1)
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    image1 = img1[cut_pixel:height1]

    img2 = cv2.imread(file2)
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    image2= img2[cut_pixel:height2]

    im_v =cv2.vconcat([image1,image2])
    cv2.imwrite(save_path, im_v)