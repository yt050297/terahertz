from pyOptoSigma import *

side_pixel = 10

def __for_predict_imaging():
    stages = Session(Controllers.SHOT_302GS)     # specify your stage controller.
    stages.append_stage(Stages.SGSP26_150)    # add your stage accordingly.
    stages.connect('COM3')                           # connect to a serial port.
    #stages.connect('/dev/ttyS4')
    #print('ステージが稼働状態かどうか:{}'.format(stages.is_ready()))
    #print(Stages(is_linear_stage()))   #stageがlinerかどうか

    stages.initialize()      #初期位置にする
    #stages.set_speed(1, S=1000, F=1000, R=100)
    #S, F, R : int, tuple, or list
    #Speed parameters of each stage.
    #S: the slowest speed, F: the fastest speed, R: acceleration and deceleration time.

    #stages.move()
    #stages.move()

    #stages.set_origin()
    #stages.reset()     #ステージを初期位置に戻す
    #stages.get_position()     #ステージの現在位置の取得
    # for i in range(side_pixel):
    #     stages.move(amount=1000, wait_for_finish=True)    # translate 1 milimeter. (1000 micro-meter)
    #     time.sleep(1) #一秒待つ


    #サンプルプログラム
    stages.set_speed(1, 1000, 100000, 500)
    #stages.set_origin()
    #stages.jog()
    #time.sleep(10)
    #stages.move(amount=1000000)
    stages.move(amount=45000, wait_for_finish=True)
    #stages.move(amount=90000, wait_for_finish=True, absolute=True)
    stages.set_origin()
    stages.jog()
    time.sleep(10)
    stages.stop()
    stages.get_position()
    stages.reset(wait_for_finish=True)
    stages.move(amount=-90000, wait_for_finish=True, absolute=True)

if __name__ == '__main__':
    __for_predict_imaging()
