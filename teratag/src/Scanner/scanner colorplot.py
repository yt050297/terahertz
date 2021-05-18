import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import math

dir_path = '/Users/kawaselab/PycharmProjects/scanner/20200701'
file_name = 'non_normalization_kyuu_vertical_raw.csv'
file = os.path.join(dir_path,file_name)
yy = np.arange(1,257,1)
zz = np.arange(1,513,1)
#print(yy)

df = np.loadtxt(file,delimiter=',',dtype = float,skiprows=2)
#df = pd.read_csv(dir_path, engine='python', header=None, skiprows=[0, 1])
print(df)

plt.pcolor(zz,yy,df)
plt.colorbar()
#plt.clim(0,0.001)
plt.axis('tight')
plt.xlabel('scanner_width[pixel]')
plt.ylabel('scanner_height[pixel]')
plt.show()



