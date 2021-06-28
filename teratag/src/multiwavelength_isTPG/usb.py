import serial

ser = serial.Serial()
ser.baudrate = 9600
ser.port = 3 # COM3->2,COM5->4
ser.open()
ser.write("Hello")