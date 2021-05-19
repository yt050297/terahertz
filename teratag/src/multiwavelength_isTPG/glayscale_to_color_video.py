#ライブラリインポート
import cv2
import numpy as np
import os
import glob

#カラーかグレースケールか選択
image_color = 'RGB'

colormap_table_count = 0
colormap_table = [
    ['COLORMAP_JET',  cv2.COLORMAP_JET],
    ['COLORMAP_AUTUMN',     cv2.COLORMAP_AUTUMN],
    ['COLORMAP_WINTER',  cv2.COLORMAP_WINTER],
    ['COLORMAP_RAINBOW', cv2.COLORMAP_RAINBOW],
    ['COLORMAP_OCEAN',   cv2.COLORMAP_OCEAN],
    ['COLORMAP_SUMMER',  cv2.COLORMAP_SUMMER],
    ['COLORMAP_SPRING',  cv2.COLORMAP_SPRING],
    ['COLORMAP_COOL',    cv2.COLORMAP_COOL],
    ['COLORMAP_HSV',     cv2.COLORMAP_HSV],
    ['COLORMAP_PINK',    cv2.COLORMAP_PINK],
    ['COLORMAP_HOT',     cv2.COLORMAP_HOT],
]

#動画のディレクトリ指定
dir_path = 'C:/Users/yt050/Desktop/grayscale_video/*.mp4'
#保存するディレクトリを指定
save_path = 'C:/Users/yt050/Desktop/grayscale_video/converted_to_color_video'
os.makedirs(save_path, exist_ok=True)
file_list = glob.glob(dir_path)

for i, file in enumerate(file_list):
        #ベース動画の読み込み
        print('実行中のファイル名 : {}'.format(file))
        save_filename1 = os.path.splitext(os.path.basename(file))[0]
        save_filename2 = save_filename1 + '_converted_to_color.mp4'
        save_filename = os.path.join(save_path, save_filename2)
        print('保存先のファイル名 : {}'.format(save_filename))

        #動画の基本情報を取得
        cam = cv2.VideoCapture(file)
        fps = cam.get(cv2.CAP_PROP_FPS)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        print('fps{}'.format(fps))
        print('height{}'.format(height))
        print('width{}'.format(width))

        # 形式はMP4Vを指定
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_filename, int(fourcc), fps, (int(width), int(height)))
        print('push q to quit')

        # 最初の1フレームを読み込む
        if cam.isOpened() == True:
            ret, frame = cam.read()
        else:
            ret = False

        while ret:
            # 読み込んだフレームを書き込み
            out.write(frame)
            ret, frame = cam.read()
            if ret == False:
                break
            if image_color == 'RGB':
                # 疑似カラーを付与
                apply_color_map_image = cv2.applyColorMap(frame, colormap_table[colormap_table_count % len(colormap_table)][1])
                #convert_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #resized_img = cv2.resize(apply_color_map_image, (width // 4, height // 4))

                #動画の表示
                cv2.imshow('image',apply_color_map_image)

            elif image_color == 'GRAY':
                apply_color_map_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = apply_color_map_image

            k = cv2.waitKey(1)
            if k == ord('q'):  # qを押すと停止する。
                break

        # 終了時の動作
        print('completed')
        cam.release()
        cv2.destroyAllWindows()
