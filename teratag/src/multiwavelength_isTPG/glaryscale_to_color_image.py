#ライブラリインポート
import cv2
import numpy as np
import os
import glob

#画像のディレクトリ指定
dir_path = 'C:/Users/yt050/Desktop/tag_time_scare/h/*.png'
#保存するディレクトリを指定
save_path = 'C:/Users/yt050/Desktop/tag_time_scare/h/converted_to_color_image'
os.makedirs(save_path, exist_ok=True)


file_list = glob.glob(dir_path)

for i, file in enumerate(file_list):
        #ベース画像の読み込み
        pic=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        print('実行中のファイル名 : {}'.format(file))
        save_filename1 = os.path.splitext(os.path.basename(file))[0]
        save_filename2 = save_filename1 + '_converted_to_color.png'
        save_filename = os.path.join(save_path, save_filename2)
        #print('保存先のファイル名 : {}'.format(save_filename))

        #疑似カラー化_JET
        pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_JET)
        #cv2.imwrite('pseudo_color_jet.jpg',np.array(pseudo_color))
        '''
        #疑似カラー化_HOT
        pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HOT)
        cv2.imwrite('pseudo_color_hot.jpg',np.array(pseudo_color))
        #疑似カラー化_HSV
        pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_HSV)
        cv2.imwrite('pseudo_color_hsv.jpg',np.array(pseudo_color))
        #疑似カラー化_RAINBOW
        pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_RAINBOW)
        '''
        #画像のサイズ変更、表示のため
        height = pseudo_color.shape[0]
        width = pseudo_color.shape[1]
        resized_img = cv2.resize(pseudo_color,(width//4,height//4))
        cv2.imshow(save_filename2, resized_img)
        cv2.waitKey(0)
        #画像の保存
        cv2.imwrite(save_filename, np.array(pseudo_color))