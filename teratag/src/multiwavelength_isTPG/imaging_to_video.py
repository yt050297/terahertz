import glob
import cv2 as cv
import os

def create_movie(dir_path):
    output = dir_path + '/video.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    for photo_name in sorted(glob.glob(dir_path + '/*.jpg')):
        im = cv.imread(photo_name)
        height, width, layers = im.shape
        size = (width, height)

        outfh = cv.VideoWriter(output, fourcc, 24, size)
        outfh.write(im)

    outfh.release()

if __name__ == '__main__':
    save_path  = 'C:/Users/yt050/Desktop/imaging_to_video'
    os.makedirs(save_path, exist_ok=True)
    create_movie(save_path)