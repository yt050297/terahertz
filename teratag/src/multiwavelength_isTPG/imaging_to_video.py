import sys
import cv2

save_path = 'C:/Users/yt050/Desktop/saveimaging/predict_imaging'
dir_path = 'C:/Users/yt050/Desktop/saveimaging/second_try/None'
# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter(save_path + '/None_video_100fps.mp4', fourcc, 100.0, (800, 800))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i in range(0, 1600):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread(dir_path + '/{}.jpg'.format(i))
    height, width, layers = img.shape
    size = (height, width)

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(i)
    print(size)

video.release()
print('written')