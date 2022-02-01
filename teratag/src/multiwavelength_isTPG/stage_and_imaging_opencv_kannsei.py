import sys
sys.path.append('../../')
from lib import AutoPolarizer
import numpy as np
import time
import cv2
import  random

def write_backgrand_imaging(height, width):
    img = np.zeros((height, width, 3), np.uint8)
    return img

def write_onepixel_imaging(img, fromleft, fromupper, color, width):
    img = cv2.rectangle(img, (fromleft, fromupper), (fromleft + width, fromupper + width), color, -1)
    return img

def color_append(pre_num, imaging, fromleft, fromupper, pixel_size):
    if pre_num == 4:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 255, 255),pixel_size)
    elif pre_num == 0:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255), pixel_size)
    elif pre_num == 1:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255, 0), pixel_size)
    elif pre_num == 2:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 0, 0), pixel_size)
    elif pre_num == 3:
        imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255, 255), pixel_size)

    return imaging

def main():
    font = cv2.FONT_HERSHEY_PLAIN
    fontsize = 6
    samplename_position_x = probability_position_x = 90
    samplename_position_y = 120
    probability_position_y = 220
    x_move = 800
    font_scale = 5

    ############イメージング表示用
    fromleft = 10  # 最初のピクセルの左端からの位置
    fromupper = 10  # 最初のピクセルの上端からの位置
    background_width = 1440
    background_height = 1440
    pixel_size = 20  # 移動距離＆1pixelのサイズ

    #######ここからステージ####################################################################
    print('ok')
    port = 'COM3'
    # 初期位置
    side_stage = 50000  ###1um/pulse
    # side_stage = 78000  ##横8cm
    # side_stage = 70000   #横3列
    height_stage = 0  ###1um/pulse
    # ピクセル数
    side_pixel = 40  ###通常時横5cm
    # side_pixel = 70   ###横8cm
    # side_pixel = 58 #横3列
    height_pixel = 20
    # height_pixel = 40
    # 解像度設定
    side_resolution = 1000
    height_resolution = 1000
    side_length = side_resolution * side_pixel
    #spd_min = 19000  # 最小速度[PPS]
    spd_min = 8000  # 最小速度[PPS]
    #spd_max = 20000  # 最大速度[PPS]
    spd_max = 10000  # 最大速度[PPS]
    acceleration_time = 1000  # 加減速時間[mS]
    sec = 0.8
    ref_time = 0.19
    ##yoko_sec = 0.35   #通常時
    yoko_sec = 0.44  # サンプル長い時

    ####イメージングの背景表示
    imaging = write_backgrand_imaging(background_height, background_width)
    cv2.imshow('image', imaging)

    # Y = np.zeros((height_pixel, side_pixel))

    polarizer = AutoPolarizer(port=port)

    print('setting_time')
    polarizer.set_speed(spd_min, spd_max, acceleration_time)
    time.sleep(2)

    print("Reset")
    polarizer.reset()
    time.sleep(25)
    print('初期位置設定')
    polarizer._set_position_relative(1, side_stage)
    time.sleep(25)
    polarizer._set_position_relative(2, height_stage)
    time.sleep(5)

    ##shoki
    polarizer._set_position_relative(2, height_resolution)
    time.sleep(sec)
    # i = j = 0
    n = 0
    y = 0
    # Y[i][j] = y
    # print(i, j)
    # print(pre)
    color_append(y, imaging, fromleft, fromupper, pixel_size)
    cv2.imshow('image', imaging)
    #save_video(imaging)
    print(n)
    # print('予測結果:{}'.format(y))
    n = n + 1

    for i in range(height_pixel):
        if i % 2 == 0:
            polarizer._set_position_relative(1, -side_length)  # 引数一つ目、1:一軸、2:2軸、W:両軸
            time.sleep(yoko_sec)

            for j in range(1, side_pixel):
                t1 = time.time()
                y = 0
                # Y[i][j] = y
                # print(i, j)
                fromleft = fromleft + pixel_size
                color_append(y, imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                #save_video(imaging)
                print(n)
                # print('予測結果{}'.format(y))
                n = n + 1
                t2 = time.time()
                time1 = t2 - t1
                time_chousetu = ref_time - time1
                time.sleep(time_chousetu)
                t3 = time.time()
                fps = 1 / (t3 - t1)
                print(fps)

        else:
            polarizer._set_position_relative(1, side_length)  # 引数一つ目、1:一軸、2:2軸、W:両軸
            time.sleep(yoko_sec)
            # time.sleep(sec)
            for j in range(1, side_pixel):
                t1 = time.time()

                y = 1  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
                # Y[i][side_pixel-1-j] = y
                # print(i, side_pixel-1-j)
                fromleft = fromleft - pixel_size
                color_append(y, imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                #save_video(imaging)
                print(n)
                # print('予測結果:{}'.format(y))
                n = n + 1
                t2 = time.time()
                time1 = t2 - t1
                time_chousetu = ref_time - time1
                time.sleep(time_chousetu)
                t3 = time.time()
                fps = 1 / (t3 - t1)
                print(fps)

        time.sleep(sec)
        polarizer._set_position_relative(2, height_resolution)
        time.sleep(sec)

        fromupper = fromupper + pixel_size

        if i % 2 == 0:
            y = 2
            # Y[i+1][j] = y
            # print(i+1, j)
            # fromleft = fromleft + pixel_size
            color_append(y, imaging, fromleft, fromupper, pixel_size)
            cv2.imshow('image', imaging)
            #save_video(imaging)
            print(n)
            # print('予測結果:{}'.format(y))
            n = n + 1


        else:
            y = 3  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
            # Y[i+1][side_pixel-1-j] = y
            # print(i+1, side_pixel-1-j)
            # fromleft = fromleft + pixel_size
            color_append(y, imaging, fromleft, fromupper, pixel_size)
            cv2.imshow('image', imaging)
            #save_video(imaging)
            print(n)
            # print('予測結果:{}'.format(y))
            n = n + 1

    time.sleep(1)
    #self.video_finish()
    polarizer.stop()

    # cv2.imshow('image',imaging)
    #self.cam_manager.stop_acquisition()
    print('Stopped Camera')


'''
    side_pixel = 44
    height_pixel = 44

    font = cv2.FONT_HERSHEY_PLAIN
    fontsize = 1.3
    fromleft = 10 #最初のピクセルの左端からの位置
    fromupper = 10#最初のピクセルの上端からの位置
    background_width = 1440
    background_height = 1440
    pixel_size = 20 #移動距離＆1pixelのサイズ
    #####ここから帰る必要あり
    fps=10

    pre_name = 'Predict Tag : '
    result_imaging = 'C:/Users/yt050/Desktop/saveimaging/pixel_imaging.mp4'


    print('ok')
    port = 'COM3'
    side_pixel = 22
    height_pixel = 22
    side_stage = 20000 + 30000  ###4um/pulse
    height_stage = 0  ###1um/pulse
    side_resolution = 2000
    height_resolution = 2000
    spd_min = 19000  # 最小速度[PPS]
    spd_max = 20000  # 最大速度[PPS]
    acceleration_time = 1000  # 加減速時間[mS]
    sec = 1

    X = np.zeros((height_pixel, side_pixel))

    polarizer = AutoPolarizer(port=port)

    # 形式はMP4Vを指定
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # print('fourcc{}'.format(fourcc))

    # 出力先のファイルを開く
    out = cv2.VideoWriter(result_imaging, int(fourcc), fps, (background_width, background_height))

    print('setting_time')
    polarizer.set_speed(spd_min, spd_max, acceleration_time)
    time.sleep(2)

    print("Reset")
    polarizer.reset()
    time.sleep(15)
    print('初期位置設定')
    polarizer._set_position_relative(1, side_stage)
    time.sleep(12)
    polarizer._set_position_relative(2, height_stage)
    time.sleep(5)


    imaging = write_backgrand_imaging(background_height,background_width)


    cv2.imshow('image',imaging)
    #cv2.waitKey(5000)



    ##shoki
    #polarizer._set_position_relative(2, height_resolution)
    #time.sleep(0.5)
    i=j=0
    a = 3
    X[i][j] = a
    print(i, j)
    color_append(a, imaging, fromleft, fromupper, pixel_size)
    cv2.imshow('image', imaging)

    for i in range(height_pixel):
        if i % 2 == 0:
        for j in range(1,side_pixel):
            
                #polarizer._set_position_relative(1, -side_resolution)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                #time.sleep(sec)
                a = 1
                X[i][j] = a
                print(i, j)
                fromleft = fromleft + pixel_size
                color_append(a, imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                #cv2.waitKey(10)
                out.write(imaging)

            else:
                #polarizer._set_position_relative(1, side_resolution)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                #time.sleep(sec)
                # a = np.random.randint(0, 2)
                a = 2
                X[i][side_pixel - 1 - j] = a
                print(i, side_pixel - 1 - j)
                fromleft = fromleft - pixel_size
                color_append(a, imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                cv2.waitKey(10)
                out.write(imaging)

        polarizer._set_position_relative(2, height_resolution)
        time.sleep(sec)
        fromupper = fromupper + pixel_size

        if i % 2 == 0:
            a = 3
            X[i + 1][j] = a
            print(i + 1, j)
            color_append(a, imaging, fromleft, fromupper, pixel_size)
            cv2.imshow('image', imaging)
            cv2.waitKey(10)
            out.write(imaging)
        else:
            a = 3
            X[i + 1][side_pixel - 1 - j] = a
            print(i + 1, side_pixel - 1 - j)
            color_append(a, imaging, fromleft, fromupper, pixel_size)
            cv2.imshow('image', imaging)
            cv2.waitKey(10)
            out.write(imaging)

    print('completed')
    #cv2.destroyAllWindows()
'''
if __name__ == "__main__":
    main()