import numpy as np
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split_madebymitsuhashi,decide_test_number,decide_test_number_onehot
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb,dnn_classification
from lib import colorcode,concentration_colorcode,colorcode_new
from keras.utils import np_utils

######遮蔽物のたびに偏光してください
width = 2
length = 4
test_number = 2
pca_third_argument = 1 # PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。
class_number = 4
from_frequency = 1.0
to_frequency = 1.7
frequency_list = []
#x_all=[]
########ここまで

base_dir = r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect/reflect'

class Read:
    def __init__(self):
        #######測定物の度に変更して下さい
        self.inten_or_trans_or_reflect = 2  # 0の時強度、1の時透過率、2の時反射率
        self.last_num = 6  # 最後の種類の使用するファイル数
        self.add = 1  # フォルダを新しく追加した場合そのフォルダの数　1つの場合は1
        ########ここまで
        self.last_type = class_number  # 使用する種類
        #self.date_dir = date_dir
        #self.shielding_material = shielding_material
        #self.sensitivity = sensitivity
        self.base_dir = base_dir
        self.from_frequency = from_frequency
        self.to_frequency = to_frequency
        self.frequency_list = frequency_list  # 周波数を指定しない場合は空にして下さい。
        self.file_name_list = []  # filenameの保持
        self.type_name_list = []  # 試薬及び厚みの保持
        self.concentration_color = [0,0,0] #初期の色情報
        self.y_all = []
        self.y_all_dnn = []
        self.flag = 0
        self.flag2 = 0

#カラーコードのタグの数width=4,length=4の場合16個のタグに対応

    def read(self):
        global x_all
        global concentration_color_type
        concentration_color_type = []  # 試薬ごとに色情報を格納

        for i in range(1,self.last_type+1):
            #ここで厚みの選択及び糖
            #i = i*0.5
            self.type_name_list.append(i)

            ###カラーコード可視化用
            #######測定物の度に変更して下さい
            concentration_interval = 0.1 # 濃度間隔 0.1_10%
            type1 = 1  # Al(OH)3とラクトース
            type2 = 2  # ラクトースとマルトース
            type3 = 3  # マルトースとAl(OH)3
            if i == 1:
                self.concentration_color[0] = int(255)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(0)
            elif i == 2:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(255)
                self.concentration_color[2] = int(0)
            elif i == 3:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(255)
            elif i == 4:
                self.concentration_color[0] = int(255)
                self.concentration_color[1] = int(255)
                self.concentration_color[2] = int(0)
            ########ここまで

            if self.flag2 == 0:
                concentration_color_type = np.array([self.concentration_color])
                self.flag2 += 1

            else:
                concentration_color_type = np.append(concentration_color_type,
                                                     np.array([self.concentration_color]), axis=0)

            ###ファイル読み込み続き
            for j in range(1,self.last_num+1):
                try:
                    x = allread(self.inten_or_trans_or_reflect,i,j,self.last_type,self.last_num,self.from_frequency,
                                self.to_frequency,self.frequency_list).Frequency_trans_reflect_TDS(self.base_dir + '/' + str(i) + '/' + str(j) + '.txt',
                            self.base_dir + '/' + str(i) + '/ref/ref.txt',self.from_frequency,self.to_frequency)

                    self.file_name_list.append(j)

                    if self.flag == 0:
                        x_all = x
                        self.flag += 1

                    else:
                        x_all = np.append(x_all, x, axis=0)


                    #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                    self.y_all.append(i)
                    self.y_all_dnn.append(i-1)
                except FileNotFoundError as e:
                    print(e)
        #print(concentration_color_type)
        return x_all, self.y_all, self.file_name_list, self.type_name_list, concentration_color_type, self.y_all_dnn

'''
###昔のやつ
#mainデータを読み込む。
for i in range(1,last_type+1):
    #i = i*0.5
    for j in range(1,last_num+1):
        try:
            x = allread(method,i,j,last_type,last_num,from_frequency,to_frequency,frequency_list).Frequency_trans_reflect_TDS(r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect/reflect/{}/{}.txt'.format(i,j),
                                                                   r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect/reflect/{}/ref.txt'.format(i),from_frequency,to_frequency)
            if flag == 0:
                x_all = x
                flag += 1
            else:
                x_all = np.append(x_all, x, axis=0)
            #i*2いる？？？？これで識別がうまくいってない？？
            y_all.append(i*2)
        except FileNotFoundError as e:
            print(e)
#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
#train_x,train_y,test_x,test_y = train_test_split_madebymitsuhashi(x_all,y_all,4)
train_x,train_y,test_x,test_y = decide_test_number(x_all,y_all,test_number)
'''

read =Read()

x_all, y_all, file_name_list, type_name_list, concentration_color_type, y_all_dnn = read.read()
train_x,train_y,test_x,test_y = decide_test_number(x_all,y_all,test_number)

#referenceのカラーコード
#colorcode(test_y,width,length)
colorcode_new(test_y, width, length, concentration_color_type)
#SVM
print('\nSVM')
best_pred=svm(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)
colorcode_new(best_pred, width, length, concentration_color_type)
#k近傍法
print('\nk近傍法')
best_pred=kNN(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)
colorcode_new(best_pred, width, length, concentration_color_type)
# PCA-SVM
print('\nPCA-SVM')
transformed, targets = pCA(x_all, y_all,pca_third_argument, file_name_list, type_name_list, concentration_color_type)
train_x_pca,train_y_pca,test_x_pca,test_y_pca = decide_test_number(transformed,targets,test_number)
best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
#colorcode(best_pred, width, length)
colorcode_new(best_pred, width, length, concentration_color_type)

## ここからDNN
Y_ = np_utils.to_categorical(np.array(y_all_dnn), class_number)
#train_x,train_y,test_x,test_y = train_test_split(x_all,Y_,1)#trainを選択する場合
train_x,train_y,test_x,test_y = decide_test_number_onehot(x_all,Y_,test_number)#testを選択する場合
best_pred, probability = dnn_classification(train_x, train_y, test_x, test_y, class_number, base_dir, from_frequency, to_frequency, frequency_list)
print('\nDNN')
print('best_pred:{}'.format(best_pred))
print('probability:{}'.format(probability))
print('average_probability:{}'.format(sum(probability)/len(probability)))
colorcode_new(best_pred, width, length, concentration_color_type)
