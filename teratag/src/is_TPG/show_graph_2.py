import pandas as pd
import os
import sys
sys.path.append('../../')
from lib import ChangeTransmittance
from lib import ReadFile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

def main():
    #ディレクトリ '/dir_path/sensivity/folder_num/file_num.txt' '/dir_path/ref_file'
    dir_path = 'C:/Users/kawaselab/PycharmProjects/tds/20200904/shatihata/CNT/afterarai'
    file_name = 'cnt_trans.txt'
    ref_file = 'ref_after.txt'
    title = 'title'
    tilte_tds_trans = ''
    tilte_tds_absorp = 'ゴムシートの電波吸収特性（透過）'

    ref = os.path.join(dir_path,ref_file) #絶対パスにする
    file = os.path.join(dir_path,file_name)


    def transmittance(file,ref):
        df_ref = pd.read_table(ref, engine='python', index_col=0, header=None)
        flag = True
        df = pd.read_table(file, engine='python', index_col=0, header=None)
        basename = os.path.basename(file)
        root, ext = os.path.splitext(basename)
        df.columns = [root]
        df.iloc[:, 0] = df.iloc[:, 0] / df_ref.iloc[:, 0]
        if flag == True:
            df = df
            flag = False

        else:
            df = df.append(self.df,sort=True)

        x_axis = 'frequency[THz]'
        y_axis = 'transmittance'
        df.plot(colormap='tab20', legend=False)
        plt.xlim(0.8, 1.6)
        plt.ylim(0, 1.2)
        # plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=14)
        # plt.xticks(np.arange(1.2, 1.6, 0.04))
        # plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0,fontsize=10)
        plt.title('title')
        plt.show()
        plt.close()




    def intensity(measurement,i):
        df = ReadFile().read_file_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'intensity[mV]'
        df.plot(colormap='tab20', legend=False, grid = False)
        #pointa = 1.0
        #pointb = 1.2
        #pointc = 1.6
        first = 0.8
        last = 1.6
        #plt.xticks([pointa,pointb,pointc,last])
        plt.xlim(first, last)
        plt.ylim(0, 25)
        #plt.vlines(pointa, 0, 25, "red", linestyles='dashed', linewidth=1)
        #plt.vlines(pointb, 0, 25, "red", linestyles='dashed',linewidth = 1)
        #plt.vlines(pointc, 0, 25, "red", linestyles='dashed', linewidth=1)
        #plt.xticks(np.arange(1.0, 1.8, 0.02))
        # plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)

        plt.tick_params(labelsize=18)
        plt.legend(fontsize=12)
        plt.title('{}'.format(i))
        plt.show()
        plt.close()


    def tds_transmittance(file,ref):
        df_ref = pd.read_table(ref, engine='python',
                               names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
        df = pd.read_table(file, engine='python',
                                names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
        basename = os.path.basename(file)
        root, ext = os.path.splitext(basename)

        df.iloc[:, 3] = df.iloc[:, 3] / df_ref.iloc[:, 3]
        df = df.iloc[:, [2, 3]]


        df = df.set_index('Frequency')
        # print('self.df')
        # print(self.df)
        df.columns = [root]

        x_axis = 'frequency[THz]'
        y_axis = 'transmittance'
        #df.plot(colormap='tab20',legend = False)
        df.plot(colormap='tab20')
        plt.xlim(0.05,1.8)
        plt.ylim(0, 0.01)
        #plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=14)
        #plt.xticks(np.arange(1.2, 1.6, 0.04))
        #plt.legend(fontsize=12)
        plt.title('title')
        plt.show()

    def tds_absorption(file,ref):
        df_ref = pd.read_table(ref, engine='python',
                               names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
        df = pd.read_table(file, engine='python',
                                names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
        basename = os.path.basename(file)
        root, ext = os.path.splitext(basename)

        df.iloc[:, 3] = df.iloc[:, 3] / df_ref.iloc[:, 3]
        df = df.iloc[:, [2, 3]]
        print(df)
        # 透過率→dB
        for i in range(len(df)):
            df[i,2] = 10 * (np.log10(df[i,2] / 100))

        print(df)
        df = df.set_index('Frequency')
        # print('self.df')
        # print(self.df)
        df.columns = [root]

        x_axis = 'frequency[THz]'
        y_axis = '電波吸収(dB)'
        #df.plot(colormap='tab20',legend = False)
        df.plot(colormap='tab20')
        plt.xlim(0.05,1.8)
        plt.ylim(0, 0.01)
        #plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=14)
        #plt.xticks(np.arange(1.2, 1.6, 0.04))
        #plt.legend(fontsize=12)
        plt.title(tilte_tds_absorp)
        plt.show()


#ここからメイン
    #transmittance(ref, measurement,i)
    #intensity(measurement,i)
    tds_transmittance(file,ref)
    tds_absorption(file,ref)




if __name__ == '__main__':
    main()