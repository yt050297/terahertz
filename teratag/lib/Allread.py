import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import rcParams
from sklearn import preprocessing
import sys
from scipy import interpolate
sys.path.append('../')
from lib.change_db import change_db

rcParams.update({'figure.autolayout': True})
thickness = ''
sample_init = 0


class allread:

    def __init__(self,method = 0,type = 3,sample = 3,last_type = 3,last_num = 3,first = 0.8,last = 2.6,frequency_list = []):
        #self.df = pd.read_table(file, engine='python')
        #self.file = file
        if method == 0:
            y_axis = 'Intensity (a.u.)'
        elif method == 1:
            y_axis = 'Transmittance'
        elif method == 2:
            y_axis = 'Reflectance'
        self.method = y_axis
        self.type = type
        self.thickness = str(type)
        self.sample = sample
        self.last_type = last_type
        self.last_num = last_num
        self.frequency_list = frequency_list
        self.first_freq = first
        self.last_freq = last
        #self.first_freq = first
        #self.last_freq = last
# ファイルを読み込む。
# 周波数全て利用
    def Time_intensity(self,file):
        self.df = pd.read_table(file, engine='python', names=('時間', '電場', '周波数', '反射率', '位相'))
        self.file = file

        self.graph_Time_intensity_everymm()
        x_list = []

        for j in self.df.iloc[:,1]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all

    def Frequency_trans_reflect_TDS(self,file,ref,first,last):
        self.df = pd.read_table(file, engine='python', names=('Time', 'Intensity', 'Frequency', 'Fre_intensity', 'Isou'))
        self.file = file
        self.first_freq = first
        self.last_freq = last
        flag = 0
        x_list = []

        df_ref = pd.read_table(ref, engine='python', names=('Time', 'Intensity', 'Frequency', 'Fre_intensity', 'Isou'))
        #ここで強度を透過率に変化
        self.df.iloc[:,3] = self.df.iloc[:,3]/df_ref.iloc[:,3]
        df_polygonal = self.df.iloc[:, [2, 3]]
        df_polygonal = df_polygonal.set_index('Frequency')
        #print(df_polygonal)
        for i,j  in enumerate(df_polygonal.index):
            if flag == 0:
                if j >= first:
                    first_index = i
                    flag = 1
            elif flag == 1:
                if j >= last:
                    last_index = i
                    flag = 2
        df_polygonal = df_polygonal.iloc[first_index:last_index]
        #df_polygonal = self.min_max_normalization_TDS(df_polygonal)
        #self.graph_Frequency_trans_reflect()
        self.graph_Frequency_trans_reflect_everymm(df_polygonal)
        #0.2~2THzを見ている。
        for j in df_polygonal.iloc[:,0]:
            x_list.append(j)
            #print(j)
        x_all =  np.array([x_list])

        return x_all

    def Frequency_trans_reflect_TDS_fre_trans_excel(self,file,first,last):
        self.df = pd.read_csv(file, engine='python', skiprows = 1, names=('Frequency','Reflectance'))
        #print(self.df)
        self.file = file
        self.first_freq = first
        self.last_freq = last
        flag = 0
        x_list = []

        self.df = self.df.set_index('Frequency')

        for i, j in enumerate(self.df.index):
            if flag == 0:
                if j >= first:
                    first_index = i
                    flag = 1
            elif flag == 1:
                if j >= last:
                    last_index = i
                    flag = 2
        self.df = self.df.iloc[first_index:last_index]
        #print(first_index)
        #print(self.df.index)

        self.frequency_list = self.df.index
        #print(len(self.frequency_list))
        frequency_numpy = np.array(self.frequency_list)
        #print(self.frequency_list)
        self.df = self.min_max_normalization_TDS(self.df)
        # self.graph_Frequency_trans_reflect()
        self.graph_Frequency_trans_reflect_everymm(self.df)
        # 0.2~2THzを見ている。
        for j in self.df.iloc[:, 0]:
            x_list.append(j)
            # print(j)
        x_all = np.array([x_list])

        return x_all, frequency_numpy, first_index

    def Frequency_trans_reflect_TDS_fre_trans_excel_cut_fre(self,file,feature_fre_list):
        self.df = pd.read_csv(file, engine='python', skiprows = 1, names=('Frequency','Reflectance'))
        pd.set_option('display.max_columns',100)
        pd.set_option('display.max_rows',500)
        #print(self.df)
        self.file = file
        self.feature_fre_list = feature_fre_list
        flag = 0
        x_list = []

        self.df_extract = self.df.iloc[self.feature_fre_list, :]

        self.df_extract = self.df_extract.set_index('Frequency')
        #print(self.df_extract)

        self.frequency_list = self.df_extract.index
        frequency_numpy = np.array(self.frequency_list)
        #print(self.frequency_list)
        self.df_extract = self.min_max_normalization_TDS(self.df_extract)
        # self.graph_Frequency_trans_reflect()
        #self.graph_Frequency_trans_reflect_everymm(self.df_extract)
        # 0.2~2THzを見ている。
        for j in self.df_extract.iloc[:, 0]:
            x_list.append(j)
            # print(j)
        x_all = np.array([x_list])

        return x_all, frequency_numpy


    def Frequency_Intencity_is_TPG(self, file, first, last):

        self.df = pd.read_table(file, engine='python', index_col=0)
        self.file = file
        self.first_freq = first
        self.last_freq = last
        x_list = []
        self.df = self.df[first:last]

        print(self.df)
        self.min_max_normalization()


        #self.graph_Frequency_trans_reflect_is_TPG()
        #print(self.df)
        for j in self.df.iloc[:, 0]:
            x_list.append(j)

        # [1.12, 1.23, 1.3, 1.36, 1.45, 1.55, 1.6]
        x_all = np.array([x_list])

        return x_all


    def Prepare_Machine_Learning(self,file,ref):
        self.df = pd.read_table(file, engine='python',index_col=0)
        self.file = file
        x_list = []

        df_ref = pd.read_table(ref, engine='python',index_col=0)
        #print(df_ref)

        #ここで強度を透過率に変化
        self.df.iloc[:,0] = self.df.iloc[:,0]/df_ref.iloc[:,0]
        self.df = self.df[first:last]
        #self.Frequency_trans_reflect_is_TPG_FFT()
        self.min_max_normalization()
        #self.graph_Frequency_trans_reflect_is_TPG()
        self.graph_Frequency_trans_reflect_is_TPG()
        #print(self.df)
        for j in self.df.iloc[:,0]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all, df_ref

    # interpld
    def spline1(self, point):
        y = list(self.df.iloc[:, 0])
        x = list(self.df.index)
        f = interpolate.interp1d(x, y, kind="cubic")  # kindの値は一次ならslinear、二次ならquadraticといった感じに
        X = np.linspace(x[0], x[-1], num=point, endpoint=True)
        Y = f(X)
        self.df = pd.DataFrame(Y, index=X)


    # Akima1DInterpolator
    def spline2(self,  point):
        y = list(self.df.iloc[:, 0])
        x = list(self.df.index)
        f = interpolate.Akima1DInterpolator(x, y)
        X = np.linspace(x[0], x[-1], num=point, endpoint=True)
        Y = f(X)
        self.df = pd.DataFrame(Y, index=X)


    # splprep
    def spline3(self,  point, deg):
        y = list(self.df.iloc[:, 0])
        x = list(self.df.index)
        tck, u = interpolate.splprep([x, y], k=deg, s=0)
        u = np.linspace(0, 1, num=point, endpoint=True)
        spline = interpolate.splev(u, tck)
        self.df = pd.DataFrame(spline[1], index=spline[0])


    def Frequency_trans_reflect_is_TPG(self,file,ref):

        self.df = pd.read_table(file, engine='python',index_col=0,header=None)
        self.file = file
        #print(self.df)
        x_list = []

        df_ref = pd.read_table(ref, engine='python',index_col=0,header=None)
        #print(df_ref)

        #ここで強度を透過率に変化
        if self.method == 'Intensity (a.u.)':
            pass
        elif self.method == 'Transmittance' or self.method == 'Reflectance':
            self.df.iloc[:, 0] = self.df.iloc[:, 0] / df_ref.iloc[:, 0]



        if not self.frequency_list:
            self.df = self.df[self.first_freq:self.last_freq]
        else:
            self.df = self.df.loc[self.frequency_list]

        #self.Frequency_trans_reflect_is_TPG_FFT(0) #振幅スペクトルが欲しい場合はnumberを0、位相スペクトルが欲しい時はnumberを1
        #self.min_max_normalization()

        #self.df = change_db(self.df)
        #self.graph_Frequency_trans_reflect_is_TPG()

        #self.spline2(800)#補間曲線を作成

        #self.graph_Frequency_trans_reflect_is_TPG_everymm('Frequency (THz)',self.method) #グラフの表示

        #print(self.df)
        for j in self.df.iloc[:,0]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all

    def graph_Frequency_trans_reflect(self):
        #print(matplotlib.rcParams['font.family'])
        plt.style.use('ggplot')
        #font = {'family': 'meiryo'}
        #matplotlib.rc('font', **font)
        #データの処理
        df = self.df.iloc[:, [2, 3]]
        df.columns = ['frequency', self.method]
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df[18:165].plot(x='frequency', y=self.method)
        plt.ylabel(self.df)
        plt.title(self.file)
        plt.show()

        return

    def graph_Frequency_trans_reflect_everymm(self,df_polygonal):
        global thickness
        global df
        plt.style.use('ggplot')
        #df_polygonal = self.df.iloc[:, [2, 3]]
        #df_polygonal = df_polygonal.set_index('周波数')
        df_polygonal.columns = [self.file[-5]]
        if thickness != self.thickness:
            df = df_polygonal
        else:
            df = df.append(df_polygonal)


        # if self.type == 3:
        #     df.plot()
        #     plt.xticks()
        #     plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
        #     plt.ylim(0,1.0)
        #     plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.02))
        #     plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.02))
        #     plt.xlabel('Frequency[THz]')
        #     plt.ylabel(self.method)
        #
        #     if self.thickness == '0':
        #         plt.title('Maltose')
        #     if self.thickness == '1':
        #         plt.title('Al(OH)3')
        #     if self.thickness == '2':
        #         plt.title('Lactose')
        #     if self.thickness == '3':
        #         plt.title('Glucose')
        #
        #     #plt.title(self.thickness)
        #     thickness = self.thickness
        #     plt.grid(which='minor')
        #     #save_path = r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect_all/save_graph2/{}/cardboad'.format(self.type)
        #     #os.makedirs(save_path, exist_ok=True)
        #     #plt.savefig(save_path + '/figure{}.jpg'.format(self.sample))
        #     plt.show()
        #     plt.close()



        return

    def graph_Time_intensity(self):
        #print(matplotlib.rcParams['font.family'])
        plt.style.use('ggplot')
        #font = {'family': 'meiryo'}
        #matplotlib.rc('font', **font)
        #データの処理
        df = self.df.iloc[:, [0, 1]]
        df.columns = ['time', 'intensity']
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df.plot(x='time', y='intensity')
        plt.ylabel('intensity')
        plt.title(self.file)
        plt.show()

        return

    def graph_Time_intensity_everymm(self):
        global thickness
        global df
        plt.style.use('ggplot')
        df_polygonal = self.df.iloc[:, [0, 1]]
        df_polygonal = df_polygonal.set_index('時間')
        df_polygonal.columns = [self.file[-5]]
        if thickness != self.thickness:
            df = df_polygonal
        else:
            df = df.append(df_polygonal)

        df.plot()
        plt.xlabel('時間[ps]')
        plt.ylabel('電場[a.u.]')
        plt.title(self.thickness)
        thickness = self.thickness
        plt.show()
        return

    def graph_Frequency_trans_reflect_is_TPG(self):
        #plt.style.use('ggplot')
        df = self.df
        df.columns = [self.sample]
        #散布図
        df.plot(legend=None)
        #plt.grid()
        plt.xlabel('frequency[THz]',fontsize = 18)
        plt.ylabel(self.method,fontsize = 18)
        plt.title('type:'+str(self.type)+'sample:' + str(self.sample))
        plt.tick_params(labelsize=18)

        plt.show()

        return

    def graph_Frequency_trans_reflect_is_TPG_everymm(self,x,y):
        global df
        global sample_init


        if sample_init == 0 and self.sample == 1:
            sample_init = 1
            self.df.columns = [self.sample]
            df = self.df
            sample_init = 1
        elif self.sample == 1:
            #print('plot')
            df.plot(colormap='tab20',legend=None)
            plt.xlabel(x,fontsize = 28)
            plt.ylabel(y,fontsize = 28)
            #plt.xticks([0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
            plt.title(self.type-1)
            plt.tick_params(labelsize=24)
            #plt.legend(fontsize=10)
            if not self.frequency_list:
                pass
            else:
                plt.xticks(self.frequency_list)
            plt.show()
            plt.close()
            self.df.columns = [self.sample]
            df = self.df

        else:
            self.df.columns = [self.sample]
            df = df.append(self.df)
        if self.last_type == self.type and self.last_num == self.sample:
            print('lastplot')
            df.plot(colormap='tab20',legend=None)
            plt.xlabel(x,fontsize = 28)
            plt.ylabel(y,fontsize = 28)
            #plt.xticks([0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
            plt.tick_params(labelsize=24)
            #plt.legend(fontsize=10)
            plt.title(self.type)
            if not self.frequency_list:
                pass
            else:
                plt.xticks(self.frequency_list)
            plt.show()
            plt.close()

        return

    def min_max_normalization(self):
        list_index = list(self.df.index)
        x = self.df.values  # returns a numpy array
        #print(x)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.01,1.01))
        x_scaled = min_max_scaler.fit_transform(x)
        #for n in self.drange(self.first_freq,self.last_freq,0.01):
            #list_index.append(n)
        self.df = pd.DataFrame(data=x_scaled,index=list_index)
        return

    def min_max_normalization_TDS(self,df):
        list_index = list(df.index)
        x = df.values  # returns a numpy array
        #print(len(x))
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        #for n in self.drange(self.first_freq,self.last_freq,0.01):
            #list_index.append(n)
        df = pd.DataFrame(data=x_scaled,index=list_index)
        return df

    def drange(self, begin, end, step):
        n = begin
        while n  < end+0.01:
            yield n
            n += step

    def Frequency_trans_reflect_is_TPG_FFT(self,number):
        list_index = list(self.df.index)
        print(list_index)
        N = len(list_index) #サンプル数
        aliasing = N/2
        dt = round(list_index[1] - list_index[0],2) #サンプリング間隔
        t = np.arange(0, N*dt, dt) # 時間軸
        list_index = list(t)
        #freqList = np.fft.fftfreq(N, d=1.0/fs)  # 周波数軸の値を計算 fsはサンプリング周波数
        a_df = self.df.values
        # ここで一次元にする事でFFT出来るようにする。
        one_dimensional_a_df = np.ravel(a_df)
        print(one_dimensional_a_df)
        #F = np.fft.fft(one_dimensional_a_df)#フーリエ変換
        F = np.fft.ifft(one_dimensional_a_df)#フーリエ逆変換
        #print(F)
        Amp = np.abs(F) #下とおんなじ
        #Amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in F]  # 振幅スペクトル
        phaseSpectrum = [np.arctan2(int(c.imag), int(c.real)) for c in F]  # 位相スペクトル
        two_dimentional_Amp = np.reshape(Amp,(len(Amp),1))
        two_dimentional_phase = np.reshape(phaseSpectrum, (len(phaseSpectrum), 1))
        #ここで振幅スペクトルか位相スペクトルかを選ぶ。
        if number == 0:
            self.df = pd.DataFrame(data=two_dimentional_Amp[:int(aliasing)], index=list_index[:int(aliasing)])
        else:
            self.df = pd.DataFrame(data=two_dimentional_phase, index=list_index)
        print(self.df)
        return