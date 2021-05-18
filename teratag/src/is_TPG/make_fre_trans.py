import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

type_num = 1
file = []
first = 0.1
last = 2.0
flag = 0

for i in range(0,type_num):
    base_path = r'C:/Users/kawaselab/PycharmProjects/tds/tds/20210212/shatihata_gomu_hakarinaosi'
    file = sorted(glob.glob(base_path + '/{}/*.txt'.format(i)))
    ref = base_path + '/{}/ref/ref.txt'.format(i)
    save_path = base_path + '/' + '{}'.format(i) + '/'+ 'changed_to_trans'
    os.makedirs(save_path, exist_ok=True)
    file_num = len(file)
    for j in range(0, file_num):
    #for j in range(1,file_num+1):
        df = pd.read_table(file[j-1], engine='python', names=('Time', 'Intensity', 'Frequency', 'Reflectance', 'Isou'))
        df_ref = pd.read_table(ref, engine='python', names=('Time', 'Intensity', 'Frequency', 'Fre_intensity', 'Isou'))
        #print(j)
        # ここで強度を透過率に変化
        df.iloc[:, 3] = df.iloc[:, 3] / df_ref.iloc[:, 3]
        df_polygonal = df.iloc[:, [2, 3]]
        df_polygonal = df_polygonal.set_index('Frequency')
        #print(df_polygonal)
        for k,l in enumerate(df_polygonal.index):
            if flag == 0:
                if l >= first:
                    first_index = k
                    flag = 1
            elif flag == 1:
                if l >= last:
                    last_index = k
                    flag = 2
        df_polygonal = df_polygonal.iloc[first_index:last_index]
        print(df_polygonal)
        df_polygonal.to_csv(save_path + '/B52-{}.csv'.format(j))

