import sys
sys.path.append('../../')
from lib import AutoPolarizer
#from AutoPolarizer import *
import time

def main():
    print('ok')
    port = 'COM3'
    side_stage = 30000+20000  ###2um/pulse
    height_stage = 0   ###1um/pulse
    spd_min = 19000  # 最小速度[PPS]
    spd_max = 20000  # 最大速度[PPS]
    acceleration_time = 1000  # 加減速時間[mS]

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
    time.sleep(12)

def main2():
    print('ok')
    port = 'COM3'
    side_stage = -5000  ###2um/pulse

    polarizer = AutoPolarizer(port=port)

    polarizer._set_position_relative(1, side_stage)
    time.sleep(8)

def main3():
    print('ok')
    port = 'COM3'
    height_stage = 5000  ###2um/pulse

    polarizer = AutoPolarizer(port=port)

    polarizer._set_position_relative(2, height_stage)
    time.sleep(8)


if __name__ == "__main__":
    #main()
    #main2()
    main3()