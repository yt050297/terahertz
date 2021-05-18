import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#os.chdir('')

df = pd.read_csv( '/Users/kawaselab/PycharmProjects/scanner/20200701/non_normalization_kyuu_vertical.csv', header=None, skiprows=2)
#河上くん用
#df_kawakami = pd.read_table('../../../sample.txt', header=None)
a_df = df.values
#a_df_kawakami = df_kawakami.astype(int).values
#print(a_df_kawakami)

#画像の表示
plt.imshow(a_df, cmap = 'gray', vmin = 0, vmax = 0.00001, interpolation = 'none')
# => plt.imshow(img_rgb, interpolation = 'none') と同じ

plt.show()
