import time
import numpy as np
import matplotlib.pyplot as plt

side_pixel = 20
height_pixel = 20

X = np.zeros((height_pixel, side_pixel))
print(X)

fig = plt.figure(figsize=(5, 5))
plt.imshow(X, cmap='bwr')
#plt.xlim(0, side_pixel, 1)
#plt.ylim(0, height_pixel, 1)
#plt.xticks(1)
#plt.yticks(1)
plt.xlabel('sample width [pixel]',fontsize=16)
plt.ylabel('sample height [pixel]', fontsize=16)
plt.title("Plot 2D array",fontsize=16)
plt.savefig('C:/Users/yt050/Desktop/saveimaging/1.png')
plt.show()