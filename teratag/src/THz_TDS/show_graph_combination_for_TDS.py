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

def trans_log(df_trans_log,title,save_path,label):
#透過率グラフ
    df = df_trans_log
    df.plot(marker='o',label=label)

    plt.xlim(0.1,2.0)
    plt.ylim(-80,0)
    #plt.yscale('log')
    x_axis = 'Frequency[THz]'
    y_axis = '電波吸収[dB]'
    plt.xlabel(x_axis, fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.grid()
    plt.legend()
    plt.title(title,fontsize=16)
    plt.savefig(save_path + '/' + 'img2.jpg')
    #plt.show()
    #plt.close()

#ファイル指定
dir_1 = 'C:/Users/TeraHertz/PycharmProjects/untitled/20201102'
ref_1 = dir_1 + '/' + 'ref.txt'
dir_2 = 'C:/Users/TeraHertz/PycharmProjects/untitled/20210212/shatihata_gomu_hakarinaosi'
ref_2 = dir_2 + '/' + 'ref.txt'
dir_3 = 'C:/Users/TeraHertz/PycharmProjects/untitled/20210212/shatihata_gomu_hakarinaosi'
ref_3 = dir_3 + '/' + 'ref.txt'
dir_4 = 'C:/Users/TeraHertz/PycharmProjects/untitled/20201102'
ref_4 = dir_3 + '/' + 'ref.txt'

save_path = 'C:/Users/TeraHertz/PycharmProjects/untitled/20210212/shatihata_gomu_hakarinaosi' + '/' + 'graph_comp'
#保存先ディレクトリがないなら作る
os.makedirs(save_path, exist_ok=True)
#テキスト番号
i = 1
j = 1
k = 2
l = 32



file_1 = dir_1 + '/' + '{}.txt'.format(i)
file_2 = dir_2 + '/' + 'THz{}_hansha.txt'.format(j)
file_3 = dir_3 + '/' + 'THz{}_hansha.txt'.format(k)
file_4 = dir_4 + '/' + '{}.txt'.format(l)

title = 'エコソーブとゴムシートの比較(反射)'

#ファイル読み込み
df = pd.read_table(file_1, engine='python',
                   names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

df_ref = pd.read_table(ref_4, engine='python',
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
trans_log(df_trans_log,title,save_path,label='B52-1')


#ファイル読み込み
df = pd.read_table(file_2, engine='python',
                   names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

df_ref = pd.read_table(ref_1, engine='python',
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
trans_log(df_trans_log,title,save_path,label='THz-1')


#ファイル読み込み
df = pd.read_table(file_3, engine='python',
                   names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

df_ref = pd.read_table(ref_2, engine='python',
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
trans_log(df_trans_log,title,save_path,label='THz-2')


#ファイル読み込み
df = pd.read_table(file_4, engine='python',
                   names=('Time', 'Intensity', 'Frequency', 'Fre_Intensity', 'Phase'))

df_ref = pd.read_table(ref_3, engine='python',
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
trans_log(df_trans_log,title,save_path,label='エコソーブ')
