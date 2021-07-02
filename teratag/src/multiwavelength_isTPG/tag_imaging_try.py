import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random

def write_backgrand_imaging(height, width):
    img = np.zeros((height, width, 3), np.uint8)
    return img

def write_onepixel_imaging(img, fromleft, fromupper, pixel_size, color, width):
    img = cv2.rectangle(img, (fromleft, fromupper), (fromleft + pixel_size, fromupper + width), color, -1)
    return img

font = cv2.FONT_HERSHEY_PLAIN
fontsize = 1.3
fromleft = 10 #最初のピクセルの左端からの位置
fromupper = 270#最初のピクセルの上端からの位置
vertical = 150 #imagingの際の縦の長さ
background_width = 1440
background_height = 540
#####ここから帰る必要あり
fps=10
pixel_size = 50 #移動距離＆1pixelのサイズ
result_imaging = 'C:/Users/yt050/Desktop/saveimaging/tag_imaging.mp4'


imaging = write_backgrand_imaging(background_height,background_width) #タグイメージングのための背景を記入
imaging = cv2.putText(imaging,'Tag predict Imaging',(10,200),2, fontsize, (255,255,255), 2, cv2.LINE_AA) #文字記入
#video_file_basename,ext = os.path.splitext(video_file_name)

cv2.imshow('image',imaging)
#cv2.waitKey(5000)

# 形式はMP4Vを指定
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# print('fourcc{}'.format(fourcc))

# 出力先のファイルを開く
out = cv2.VideoWriter(result_imaging, int(fourcc), fps, (background_width,background_height))

for k in range(50):
    y = random.randint(0,3)

    if y == 3:
        imaging = write_onepixel_imaging(imaging,fromleft,fromupper,pixel_size,(255, 255, 255),vertical)
    elif y == 0:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size,(0, 0, 255), vertical)
    elif y == 1:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size,(0, 255, 0), vertical)
    elif y == 2:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size,(255, 0, 0), vertical)

    fromleft = fromleft + pixel_size
    cv2.imshow('image',imaging)
    cv2.waitKey(10)
    out.write(imaging)

print('completed')
#cv2.destroyAllWindows()
