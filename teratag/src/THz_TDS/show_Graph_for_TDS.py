import matplotlib.pyplot as plt
import  matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os
#import matplotlib as mpl
#print(mpl.matplotlib_fname())
#mpl.font_manager._rebuild()

def intensity(df):
    #強度グラフ
    plt.plot(df)
    plt.xlim(0.1, 2)
    #plt.ylim()
    plt.yscale('log')
    x_axis = 'frequency[THz]'
    y_axis = 'intensity'
    plt.xlabel(x_axis, fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.grid()
    plt.title('lac_1mm_1_intensity')
    plt.show()

def transmittance(df_trans):
#透過率グラフ
    df = df_trans
    plt.plot(df)

    plt.xlim(0.1, 2)
    plt.ylim(0, 1.0)
    #plt.yscale('log')
    x_axis = 'frequency[THz]'
    y_axis = 'transmittance'
    plt.xlabel(x_axis, fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.grid()
    plt.title('lac_1mm_1_trans')
    plt.show()

def trans_log(df_trans_log,title,save_path,i):
#透過率グラフ
    df = df_trans_log
    df.plot(marker='o')

    plt.xlim(0.1,2.0)
    plt.ylim(-80,0)
    #plt.yscale('log')
    x_axis = 'Frequency[THz]'
    y_axis = '電波吸収[dB]'
    plt.xlabel(x_axis, fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.grid()
    plt.title(title,fontsize=16)
    plt.savefig(save_path + '/' + 'img{}.jpg'.format(i))
    #plt.show()
    plt.close()

#ファイル指定
dir = 'C:/Users/yt050/PycharmProjects/20210519/trans'
ref = dir + '/' + 'ref.txt'
save_path = dir + '/' + 'graph'
#保存先ディレクトリがないなら作る
os.makedirs(save_path, exist_ok=True)
first = 25
last = 25

for i in range(first,last+1):
    file = dir + '/' + 'B52-{}.txt'.format(i)
    title = '試験番号：B52-{}'.format(i)

    #ファイル読み込み
    df = pd.read_table(file, engine='python',
                       names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

    df_ref = pd.read_table(ref, engine='python',
                           names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

    #print(df)
    #透過率計算
    df_trans = df
    df_trans.iloc[:, 3] = df.iloc[:, 3] / df_ref.iloc[:, 3]

    #電波吸収に変える
    df_trans_log = df_trans
    df_trans_log.iloc[:,3] = 10*np.log10(df_trans.iloc[:,3])

    #index指定
    df = df.set_index('Frequency')
    df_trans = df_trans.set_index('Frequency')
    df_trans_log = df_trans_log.set_index('Frequency')
    #行列指定
    df = df.iloc[:,2:4]
    df_trans = df_trans.iloc[:,2:4]
    df_trans_log = df_trans_log.iloc[:,2]

    # 保存先ディレクトリがないなら作る
    os.makedirs(save_path, exist_ok=True)

    #関数の実行
    #intensity(df)
    #transmittance(df_trans)
    trans_log(df_trans_log,title,save_path,i)
