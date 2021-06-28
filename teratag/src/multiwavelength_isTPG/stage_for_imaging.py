import sys
sys.path.append('../../')
from lib import AutoPolarizer
#from AutoPolarizer import *
import time
import serial


def main():
    print('ok')
    port = 'COM3'
    list = []
    side_pixel = 5
    height_pixel = 5
    #print(port)

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
    print(list)

    for i in range(height_pixel):
        for j in range(side_pixel):
            if i % 2 == 0:
                polarizer._set_position_relative(1, -500)    #引数一つ目、1:一軸、2:2軸、W:両軸
            else:
                polarizer._set_position_relative(1, 500)  # 引数一つ目、1:一軸、2:2軸、W:両軸
            time.sleep(0.6)
            #print(polarizer._get_position())
            list.append(polarizer._get_position())
            print(list)
            print(len(list))   #全ピクセル数＋１になっている

        if i == height_pixel-1:
            pass
        else:
            polarizer._set_position_relative(2, -500)


    # polarizer._set_position_relative(500)
    # time.sleep(1)
    # #print(polarizer._get_position())
    # list.append(polarizer._get_position())
    # print(list)

    time.sleep(1)


    polarizer.stop()
    #print("jog-")
    #polarizer.jog_minus()
    # time.sleep(100)
    #
    # polarizer.stop()
    # time.sleep(10)

    #print("Set speed as default")
    #polarizer.set_speed()



    #print("Set speed faster")
    #polarizer.set_speed(500, 10000, 200)



if __name__ == "__main__":
    main()