import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob
import sys
#import os
#from sklearn.model_selection import train_test_split, GridSearchCV
sys.path.append('../../')
from lib import colorcode,concentration_colorcode,colorcode_new
from lib import train_test_split_madebymitsuhashi,decide_test_number,decide_test_number_onehot
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb,dnn_classification

# Pandas のデータフレームとして表示

#データの定義
file_type = 4
first = 1.0
last = 1.7
test_number = 2
flag = 0

x_list = []
y_list = []
k = 0
f_num = []
frequency_list=[]
#all_f =[]

#以下教師データのラベルづけ
for i in range(1,file_type+1):
    #path_1 = os.chdir(r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect/reflect/{}'.format(w))

    file_list = sorted(glob.glob(r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect/reflect/{}/changed_to_trans/*.csv'.format(i)))
    #print(file_list)
    file_number = len(file_list)
    f_num.append(file_number)
    #print(file_number)
    #a = np.empty((121,file_number))
    for j in range(1,file_number+1):
        #df = pd.read_table(file_list[k], engine='python', header=None, index_col=0, sep=',')
        df = pd.read_csv(file_list[j-1], engine='python', skiprows = 1, names=('Frequency', 'Reflectance'))
        df = df.set_index('Frequency')
        #print(df)

        #df = df[first:last]#特定の周波数範囲の抜き取り
        #df = Max_Min(df)#正規化
        #print(df.iloc[:,0])

        for num,fre  in enumerate(df.index):
            if flag == 0:
                if fre >= first:
                    first_index = num
                    flag = 1
            elif flag == 1:
                if fre >= last:
                    last_index = num
                    flag = 2
        df = df.iloc[first_index:last_index]
        #print(df)
        df_np = df.values
        #print(df_np)
        # ここまで欲しいところを抜き出している過程
        if j == 1:
            x_all = df_np
            all_f = df
            #j = j + 1
            #print(x_all)
        else:
            x_all = np.append(x_all,df_np,axis = 1)
            #all_f[j] = df[0]
        y_list.append(k)
    #x_all = np.array([x_list])
    if k == 0:
        X_all = x_all
        #All_f = all_f
    else:
        X_all = np.append(X_all,x_all, axis = 1)
        #All_f = pd.concat([All_f,all_f],axis = 1)
    k = k+1
y_all = np.array(y_list)
#print(y_all)

#print(All_f.shape)
#ファイルの数
#print(f_num)

#学習データとテストデータに分割
#print(x_all)
#print(X_all.T)
#x_train, x_test, y_train, y_test = train_test_split(X_all.T, y_all, test_size=0.20)
train_x, train_y, test_x, test_y = decide_test_number(X_all.T, y_all, test_number)


randomforest(train_x, train_y, test_x, test_y, first, last, frequency_list)

# 学習
#clf = RandomForestClassifier(n_estimators=20, random_state=42)
#clf.fit(x_train, y_train)

#予測データ作成
#y_predict = clf.predict(x_test)

#正解率
#print('\n正答率')
#print(accuracy_score(y_test, y_predict))
#print(y_test)
#print(y_predict)
colorcode_new(best_pred, width, length, concentration_color_type)

#gs = GridSearchCV(RandomForestClassifier(),search_params,cv = 3, verbose True,n_jobs=-1)


#特徴量の重要度
feature = clf.feature_importances_
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})



f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']




df = pd.read_csv(file_list[k], engine='python', header=None, sep=',')
#print(int((first-0.8)*100+1))
#print(int((last-0.8)*100+2))
df = df[int((first-0.8)*100+1):int((last-0.8)*100+2)]#周波数の値を1に直したかった。なんか他に方法あるかもやけどめんどかったからこうしてる。改善してくれ。
df_np_2 = df.values
df_np2 = df_np_2[:,0]
X_all2 = np.insert(X_all.T, 0, df_np2, axis = 0)#多次元配列に一次元配列を追加する場合はappendやstackでは無理。insertを使う
print(X_all2)


#print('Feature Importances:')
#for i, feat in enumerate(iris['feature_names']):
 #   print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

#firstとlast、つまり周波数範囲に応じた１０刻みからなるリスト
i = int((last-first)*10)
freq_list = []
freq_list2 = []
for p in range(0,i+1):
    p = int(p)
    freq_list.append(10*p)
    freq_list2.append(round ( (1.1+(p/10)),3))


print(freq_list2)
#print(feature.shape)
#for i in range(0,len(feature)):
    #print('\t{0}:{1}'.format(X_all2[0,i], feature[i]))
fig = plt.figure()
plt.title('Fiture Impotance')
left = X_all2[0,:]
#print(X_all2[0,:])
X_list = X_all2[0,:].tolist()
X_list_str = [str(n) for n in X_list]
print(X_list_str)
height = feature
ax = fig.add_subplot(1,1,1)
ax.bar(range(len(feature)),height)
#ax.set_xticks([0,10,20,30,40,50,60,70])
#ax.set_xticklabels([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8], rotation=30)
ax.set_xticks(freq_list)
ax.set_xticklabels(freq_list2, rotation=30)

#グラフの表示
plt.show()


'''
plt.title('Fiture Impotance')
left = X_all2[0,:]
#print(X_all2[0,:])
X_list = X_all2[0,:].tolist()
X_list_str = [str(n) for n in X_list]
print(X_list_str)
height = feature
ax = plt.bar(range(len(feature)),height)
plt.set_xticklabels(X_all2[0,:], rotation=90)
#plt.bar(X_all2[0,:].T,height)
plt.xticks()

plt.show()

'''
'''
#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']

#特徴量の名前
label = df.columns[0:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]
'''
'''
for i in range(len(feature)):
    print(str(i + 1)) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]])
'''
'''
plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()

'''