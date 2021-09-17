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
dir_path = '/Users/yt050/PycharmProjects/yamamoto/20210917/shatihata_gomusheet/tate'
#画像の枚数
img_num = 12



'''
#強度分布をグラフ化

for i in range(0,img_num):
    #i = i * 5
    title = 'cardboard_{}'.format(i)
    file_name = '{}_1_raw.csv'.format(i)
    file = os.path.join(dir_path,file_name)
    #col = ['{0}'.format(j) for j in range(512)]
    try:
        pw_max = allread_scanner(file,title).lightsource_beamshape_smoothing()
    except FileNotFoundError as e:
        print(e)
#allread_scanner(file,title).plot_attenuation(pw_max)
'''

'''
#強度分布をイメージングで可視化
for i in range(1, img_num):
    title = 'scanner_intensity_graph'
    file_name = '{}_1_raw.csv'.format(i)
    file = os.path.join(dir_path,file_name)
    image = allread_scanner(file,title).visuallization()
'''


#分割測定からの再構成用

#分割測定用カットするピクセル数
cut_pixel1 = 157
cut_pixel2 = 87
#画像の選択
i = 1
#下の画像のシフト量（＋、ーで）
shift = -15


#分割測定からの画像の再構成
#glob.globでまとめてとってくるのもあり？？
save_path = dir_path + '/savepath'
os.makedirs(save_path, exist_ok=True)

file_name1 = '{}_2.png'.format(i)
file_name2 = '{}_1.png'.format(i)

file1 = os.path.join(dir_path,file_name1)
file2 = os.path.join(dir_path, file_name2)
print(file1)
img1 = cv2.imread(file1)
height1 = img1.shape[0]
width1 = img1.shape[1]
image1 = img1[0 : 0 + cut_pixel1]

img2 = cv2.imread(file2)
image2 = img2[cut_pixel2 : 256]
height2 = image2.shape[0]
width2 = image2.shape[1]

M = np.float32([[1,0,shift],[0,1,0]])
image2_2 = cv2.warpAffine(image2,M,(width2,height2))

im_v =cv2.vconcat([image1,image2_2])
cv2.imshow('sample1',im_v)
cv2.waitKey(1000)

#保存先ディレクトリがないなら作る
os.makedirs(save_path, exist_ok=True)
#保存
cv2.imwrite(save_path + '/THz-{}_tate.png'.format(i), im_v)
