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
file_start = 9
file_num = 9

base_path = r'C:/Users/yt050/PycharmProjects/yamamoto/TDS/torinaosi2'

ref = base_path + '/ref/ref.txt'
save_path = base_path + '/'+ 'changed_to_trans'
os.makedirs(save_path, exist_ok=True)
#file_num = len(file)
for j in range(file_start, file_num+1):
    file = base_path + '/{}_omote5.txt'.format(j)
    df = pd.read_table(file, engine='python', names=('Time', 'Intensity', 'Frequency', 'Reflectance', 'Isou'))
    df_ref = pd.read_table(ref, engine='python', names=('Time', 'Intensity', 'Frequency', 'Fre_intensity', 'Isou'))
    print(file)
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
    df_polygonal.to_csv(save_path + '/No_{}.csv'.format(j))

