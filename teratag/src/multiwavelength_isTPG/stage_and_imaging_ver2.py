import sys
sys.path.append('../../')
from lib import AutoPolarizer
#from AutoPolarizer import *
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('ok')
    port = 'COM3'
    side_pixel = 22
    height_pixel = 22
    side_stage= 20000 +30000  ###4um/pulse
    height_stage= 0 ###1um/pulse
    side_resolution = 2000
    height_resolution = 2000
    spd_min = 19000  # 最小速度[PPS]
    spd_max = 20000  # 最大速度[PPS]
    acceleration_time = 1000  # 加減速時間[mS]
    sec=1
    save_path = 'C:/Users/yt050/Desktop/saveimaging'

    X = np.zeros((height_pixel, side_pixel))

    polarizer = AutoPolarizer(port = port)

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
    #print(polarizer._get_position())
    #list.append(polarizer._get_position())
    #print(list)


    ##shoki
    polarizer._set_position_relative(2, height_resolution)
    time.sleep(sec)
    # print(polarizer._get_position())
    #list.append(polarizer._get_position())
    # print(list)
    # print(len(list))  # 全ピクセル数＋１になっている
    i=j=0
    a = 3
    X[i][j] = a
    print(i, j)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(X, cmap='bwr')
    plt.title("Plot 2D array", fontsize=16)
    plt.xlabel('sample width [pixel]', fontsize=16)
    plt.ylabel('sample height [pixel]', fontsize=16)
    # plt.colorbar()
    plt.savefig(save_path + '/{}_{}.jpg'.format(i, j))
    plt.show()


    for i in range(height_pixel):
        for j in range(1,side_pixel):
            if i % 2 == 0:
                polarizer._set_position_relative(1, -side_resolution)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                time.sleep(sec)
                a = 1
                X[i][j] = a
                print(i, j)
            else:
                polarizer._set_position_relative(1, side_resolution)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                time.sleep(sec)
                # a = np.random.randint(0, 2)
                a = 2
                X[i][side_pixel-1-j] = a
                print(i, side_pixel-1-j)
            #list.append(polarizer._get_position())
            # print(list)
            # print(len(list))   #全ピクセル数＋１になっている
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(X, cmap='bwr')
            plt.title("Plot 2D array", fontsize=16)
            plt.xlabel('sample width [pixel]', fontsize=16)
            plt.ylabel('sample height [pixel]', fontsize=16)
            # plt.colorbar()
            plt.savefig(save_path + '/{}_{}.jpg'.format(i, j))
            plt.show()

        polarizer._set_position_relative(2, height_resolution)
        time.sleep(sec)
        # print(polarizer._get_position())
        #list.append(polarizer._get_position())
        #print(list)
        #print(len(list))  # 全ピクセル数＋１になっている

        #a = np.random.randint(0, 3)
        if i % 2 == 0:
            a = 3
            X[i+1][j] = a
            print(i+1,j)
        else:
            a = 3
            X[i+1][side_pixel-1-j] = a
            print(i+1, side_pixel-1-j)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(X, cmap='bwr')
        plt.title("Plot 2D array",fontsize=16)
        plt.xlabel('sample width [pixel]', fontsize=16)
        plt.ylabel('sample height [pixel]', fontsize=16)
        #plt.colorbar()
        plt.savefig(save_path + '/{}_{}.jpg'.format(i,j))
        plt.show()

    time.sleep(1)
    polarizer.stop()




if __name__ == "__main__":
    main()