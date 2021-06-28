import numpy as np
import matplotlib.pyplot as plt
import time

X = np.zeros((5,5))
#print(X)
#list=[[0]*5]*5
for j in range(5):
    for i in range(5):
        a = np.random.randint(0,3)
        # if a == 1:
        #     a = [0.0, 0.0, 0.0]
        # if a == 2:
        #     a = [0.0, 1.0, 0.0]
        # if a == 3:
        #     a = [0.0, 0.0, 1.0]
        X[j][i]=a
        #list[j][i] = a
        #print(list)
        fig = plt.figure(figsize=(5,5))
        plt.imshow(X, cmap='bwr')
        plt.title("Plot 2D array")
        plt.show()
        #
time.sleep(1)

a=[0.0,0.0,0.0]
b=[1.0,0.0,0.0]
c=[0.0,0.0,1.0]

X=[[a,b,c],[a,b,c]]
plt.imshow(X)
plt.show()
plt.close()