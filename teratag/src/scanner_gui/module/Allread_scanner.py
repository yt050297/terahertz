import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import ListedColormap

class allread_scanner:

    def __init__(self,file,file_2,title,save_file, saveflag=0):
        self.file = file
        self.file_2 = file_2
        self.file_name = os.path.splitext(os.path.basename(file))[0]
        if len(title) == 0:
            self.title = self.file_name
        else:
            self.title = title
        self.save_file = save_file
        self.save_flag = saveflag
        x = []
        for i in range(1, 257):
            i = i * (384/256)
            x.append(i)
        self.x = x

    def lightsource_beamshape(self):
        df = pd.read_csv(self.file, engine='python', header=None, skiprows=[0, 1])
        # 通常行ラベルは勝手に番号付けされて、index_col = '0'にすると行ラベルなくなる？？インデックスコラムとは？？
        # 列ラベルは先頭一列目が使われてしまうheader = Noneを使う

        total_average = []
        sum = 0

        for j in range(0, 256):
            #時間軸方向を平均化
            for k in range(0, 512):
                if k == 0:
                    sum = df.iloc[j][k]
                else:
                    sum = sum + df.iloc[j][k]

            average = sum / 512
            # print('平均{}'.format(average))

            total_average.append(average)
        self.plot_graph(total_average)
        #print(len(total_average))

    def lightsource_beamshape_smoothing(self):
        df = pd.read_csv(self.file, engine='python', header=None, skiprows=[0, 1])
        # 通常行ラベルは勝手に番号付けされて、index_col = '0'にすると行ラベルなくなる？？インデックスコラムとは？？
        # 列ラベルは先頭一列目が使われてしまうheader = Noneを使う

        total_average = []
        list = []
        sum = 0
        flag = 0

        for j in range(0, 256):
            #時間軸方向を平均化
            for k in range(0, 512):
                if k == 0:
                    sum = df.iloc[j][k]
                else:
                    sum = sum + df.iloc[j][k]

            average = sum / 512
            # print('平均{}'.format(average))

            total_average.append(average)

        if flag == 0:
            pw_max = max(total_average)
            flag == 1
        else:
            pw_max.append(total_average)
        #print(pw_max)

        #平滑化
        for l in range(0,256):
            if l == 0 or l == 255:
                list.append(total_average[l])
            else:
                smoothing_average = (total_average[l-1] + total_average[l] + total_average[l+1])/3
                list.append(smoothing_average)

        graphdata = self.plot_graph(list)
        # print(len(total_average))

        return graphdata

    def picture_reconstruction(self,shift):

        # 分割測定からの再構成用

        # 分割測定用カットするピクセル数
        cut_pixel1 = 157
        cut_pixel2 = 87
        # 下の画像のシフト量（＋、ーで）
        #shift = -15

        # 分割測定からの画像の再構成

        img1 = cv2.imread(self.file)
        height1 = img1.shape[0]
        width1 = img1.shape[1]
        image1 = img1[0: 0 + cut_pixel1]

        img2 = cv2.imread(self.file_2)
        image2 = img2[cut_pixel2: 256]
        height2 = image2.shape[0]
        width2 = image2.shape[1]

        M = np.float32([[1, 0, shift], [0, 1, 0]])
        image2_2 = cv2.warpAffine(image2, M, (width2, height2))

        im_v = cv2.vconcat([image1, image2_2])
        cv2.imshow('sample1', im_v)
        cv2.waitKey(1000)

        # 保存先ディレクトリがないなら作る
        #os.makedirs(save_path, exist_ok=True)

        # 保存
        if self.save_flag == 0:
            cv2.imwrite(self.save_file + '/' + 'pic_reconstruction_' + self.file_name + '.png', im_v)
        else:
            pass
        cv2.destroyAllWindows()

    def plot_graph(self,y):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        #plt.rcParams['font.size'] = 17
        ax.set_title('{}'.format(self.title),fontsize=17)
        ax.set_xlabel('array_distance [mm]',fontsize=17)
        ax.set_ylabel('beam power [a.u.]',fontsize=17)
        ax.set_xticklabels([0,50,100,150,200,250,300,350,384],fontsize=17)
        ax.plot(self.x,y)
        ax.tick_params(labelsize=17)
        #ax.set_yticklabels(ax.get_yticklabels(), fontsize=17)
        #ax.imshow()
        if self.save_flag == 0:
            plt.savefig(self.save_file + '/' + 'graph_' + self.file_name + '.png')
            print('ok')
        else:
            pass

        return fig

    def plot_attenuation(self,y):
        #x_1 = list(range(1,21,1))
        plt.plot(self.x_1, y)
        plt.title('{}_attenuation'.format(self.title))
        plt.xlabel('array_distance[mm]')
        plt.ylabel('beam power[a.u.]')
        plt.xticks(1,20)
        plt.show()

    def visuallization(self, minlim=0, maxlim=0.05):
        df = np.loadtxt(self.file, delimiter=',', dtype=float, skiprows=2)
        # df = pd.read_csv(dir_path, engine='python', header=None, skiprows=[0, 1])
        print(df)
        #yy = np.arange(1, 257, 1)
        xx = np.arange(1,257,1)
        yy = np.arange(1,513,1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        c_ax = ax.pcolor(yy, xx, df)
        pp = fig.colorbar(c_ax, label='sub_THz_intensity [a.u.]', orientation="horizontal")
        pp.mappable.set_clim(minlim, maxlim)
        ax.set_title(self.title, fontsize=17)
        ax.axis('tight')
        ax.set_xlabel('scanner_move_length[a.u.]', fontsize=17)
        ax.set_ylabel('scanner_height [mm]', fontsize=17)

        '''
        plt.pcolor(yy, xx, df)
        plt.colorbar(label='sub_THz_intensity [a.u.]', orientation="horizontal")
        plt.clim(0,0.0025)
        plt.title(self.title, fontsize=17)
        plt.axis('tight')
        plt.set_xlabel('scanner_move_length[a.u.]',fontsize=17)
        plt.set_ylabel('scanner_height [mm]',fontsize=17)
        # plt.show()
        '''
        if self.save_flag == 0:
            plt.savefig(self.save_file + '/' + 'imaging_' + self.file_name + '.png')
        else:
            pass

        return fig

