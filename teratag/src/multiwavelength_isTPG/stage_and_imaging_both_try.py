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
    list = []
    side_pixel = 5
    height_pixel = 5
    flag=0
    X = np.zeros((height_pixel, side_pixel))

    polarizer = AutoPolarizer(port = port)

    print('setting_time')
    spd_min = 19000  # 最小速度[PPS]
    spd_max = 20000  # 最大速度[PPS]
    acceleration_time = 1000  # 加減速時間[mS]
    polarizer.set_speed(spd_min, spd_max, acceleration_time)
    time.sleep(2)

    print("Reset")
    polarizer.reset()
    time.sleep(6)
    #print(polarizer._get_position())
    list.append(polarizer._get_position())
    #print(list)


    for i in range(height_pixel):
        if flag == 0:
            j=0
            flag=2
        if flag == 1:
            j=1
        if flag == 2:
            j=2
        if flag == 3:
            j=3
            polarizer._set_position_relative(2, -500)

            time.sleep(0.6)
            # print(polarizer._get_position())
            list.append(polarizer._get_position())
            #print(list)
            #print(len(list))  # 全ピクセル数＋１になっている

            #a = np.random.randint(0, 3)
            a = 3
            X[i][j] = a
            print(i,j)
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(X, cmap='bwr')
            plt.title("Plot 2D array")
            plt.show()

        else:
            for j in range(side_pixel-1):
                if i % 2 == 0:
                    polarizer._set_position_relative(1, 500)    #引数一つ目、1:一軸、2:2軸、W:両軸
                    a = 1
                    X[i-1][j-1] = a
                    print(i,j)
                else:
                    polarizer._set_position_relative(1, -500)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                    #a = np.random.randint(0, 2)
                    a = 2
                    X[i-1][j+1] = a
                    print(i,j)

                time.sleep(0.6)
                list.append(polarizer._get_position())
                #print(list)
                #print(len(list))   #全ピクセル数＋１になっている

                fig = plt.figure(figsize=(5, 5))
                plt.imshow(X, cmap='bwr')
                plt.title("Plot 2D array")
                plt.show()

            polarizer._set_position_relative(2, -500)

            time.sleep(0.6)
            # print(polarizer._get_position())
            list.append(polarizer._get_position())
            #print(list)
            #print(len(list))  # 全ピクセル数＋１になっている

            #a = np.random.randint(0, 3)
            a = 3
            X[i][j] = a
            print(X)
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(X, cmap='bwr')
            plt.title("Plot 2D array")
            plt.show()

    # polarizer._set_position_relative(500)
    # time.sleep(1)
    # #print(polarizer._get_position())
    # list.append(polarizer._get_position())
    # print(list)

    time.sleep(1)


    polarizer.stop()




if __name__ == "__main__":
    main()