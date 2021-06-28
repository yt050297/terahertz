import sys
sys.path.append('../../')
from lib import AutoPolarizer
#from AutoPolarizer import *
import time
import serial


def main():
    print('ok')
    #port = serial.Serial("/dev/ttyUSB-SERIALCH340")
    #port = serial.Serial("/dev/ttyUSB4")
    #port = serial.Serial('COM3', baudrate=19200, parity=serial.PARITY_NONE)
    #port = serial.Serial(port='/dev/tty7221-7EB6')
    #port = '/dev/tty.5&2F0C7D4A&0&2'
    sport = 'COM3'
    #print(port)

    polarizer = AutoPolarizer(port = sport)

    #spd_min = 500  # 最小速度[PPS]
    #spd_max = 5000  # 最大速度[PPS]
    #acceleration_time = 200  # 加減速時間[mS]
    #polarizer.set_speed(spd_min, spd_max, acceleration_time)
    #print(polarizer._get_position())
    #polarizer.stop()
    #time.sleep(2)
    print("Reset")
    polarizer.reset()
    time.sleep(6)
    print(polarizer._get_position())
    time.sleep(2)
    polarizer._set_position_relative(5000)
    time.sleep(2)
    print(polarizer._get_position())
    time.sleep(2)
    polarizer._set_position_relative(5000)
    time.sleep(2)
    print(polarizer._get_position())
    time.sleep(2)
    print("jog+")
    polarizer.jog_plus()
    time.sleep(2)
    polarizer.stop()
    #print("jog-")
    #polarizer.jog_minus()
    # time.sleep(100)
    #
    # polarizer.stop()
    # time.sleep(10)

    #print("Set speed as default")
    #polarizer.set_speed()

    # print("Rotate +45")
    # for i in range(8):
    #     polarizer.degree += 45
    #     print("\t", polarizer.degree)
    #
    # time.sleep(1)

    #print("Set speed faster")
    #polarizer.set_speed(500, 10000, 200)

    # print("Rotate +45")
    # for i in range(8):
    #     polarizer.degree += 45
    #     print("\t", polarizer.degree)

if __name__ == "__main__":
    main()