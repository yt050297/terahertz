import numpy as np
import glob
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split_madebymitsuhashi,decide_test_number,decide_test_number_onehot
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb,dnn_classification,xgboost,lightgbm,catboost,sFs,sBs
from lib import concentration_colorcode,colorcode_new
from boruta import BorutaPy
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from keras.utils import np_utils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import rcParams

######遮蔽物のたびに偏光してください
width = 6
length = 4
test_number = 6
cut_fre = 0 #0ならfrom_toの全周波数、１ならリストの周波数選択用
feedbck = 0 #0ならなし,1ならフィードバック
wrapper_method = 4  #xgboost=0, svm=1, knn=2, pca-svm=3 randomforestとBoruta=4 lightGBM=5 catboost=6
pca_third_argument = 1 # PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。
class_number = 4
from_frequency = 1.0
to_frequency = 1.7
frequency_list = []
max_feature_list = []
#x_all=[]
########ここまで

base_dir = r'C:/Users/kawaselab/PycharmProjects/tds/siyaku_reflect_all'

class Read:
    def __init__(self):
        #######測定物の度に変更して下さい
        self.inten_or_trans_or_reflect = 2  # 0の時強度、1の時透過率、2の時反射率
        #self.last_num = 6  # 最後の種類の使用するファイル数
        self.add = 1  # フォルダを新しく追加した場合そのフォルダの数　1つの場合は1
        ##周波数選択用
        self.feature_fre_list = sorted([79, 80, 81, 82, 83, 84, 85, 89, 90, 91, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 119, 120, 121, 122, 123, 124, 125]

)
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
        self.f_num = []
        self.file_list = []

#カラーコードのタグの数width=4,length=4の場合16個のタグに対応

    def read(self):
        global x_all
        global concentration_color_type
        concentration_color_type = []  # 試薬ごとに色情報を格納

        for i in range(0,self.last_type):
            #ここで厚みの選択及び糖
            #i = i*0.5
            self.file_list = sorted(glob.glob(self.base_dir + '/{}/changed_to_trans/*.csv'.format(i)))
            #self.file_list = glob.glob(self.base_dir + '/{}/changed_to_trans/*.csv'.format(i))
            #print(self.file_list)
            self.last_num = len(self.file_list)
            #print(self.last_num)
            self.f_num.append(self.last_num)

            self.type_name_list.append(i)

            ###カラーコード可視化用
            #######測定物の度に変更して下さい
            concentration_interval = 0.1 # 濃度間隔 0.1_10%
            type1 = 1  # Al(OH)3とラクトース
            type2 = 2  # ラクトースとマルトース
            type3 = 3  # マルトースとAl(OH)3
            if i == 0:
                self.concentration_color[0] = int(255)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(0)
            elif i == 1:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(255)
                self.concentration_color[2] = int(0)
            elif i == 2:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(255)
            elif i == 3:
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
                    x, frequency_list, first_index = allread(self.inten_or_trans_or_reflect,i,j,self.last_type,self.last_num,self.from_frequency,
                                self.to_frequency,self.frequency_list).Frequency_trans_reflect_TDS_fre_trans_excel(self.file_list[j-1],self.from_frequency,self.to_frequency)
                    #print(self.file_list[j-1])
                    self.file_name_list.append(j)

                    if self.flag == 0:
                        x_all = x
                        self.flag += 1

                    else:
                        x_all = np.append(x_all, x, axis=0)


                    #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                    self.y_all.append(i)
                    self.y_all_dnn.append(i)
                except FileNotFoundError as e:
                    print(e)
            #plt.plot()

        #print(concentration_color_type)
        return x_all, self.y_all, self.file_name_list, self.type_name_list, concentration_color_type, self.y_all_dnn, frequency_list, first_index

    def read_cut_fre(self):
        global x_all
        global concentration_color_type
        concentration_color_type = []  # 試薬ごとに色情報を格納

        for i in range(0,self.last_type):
            #ここで厚みの選択及び糖
            #i = i*0.5

            self.file_list = sorted(glob.glob(self.base_dir + '/{}/changed_to_trans/*.csv'.format(i)))
            #print(self.file_list)
            self.last_num = len(self.file_list)
            #print(self.last_num)
            self.f_num.append(self.last_num)

            self.type_name_list.append(i)

            ###カラーコード可視化用
            #######測定物の度に変更して下さい
            concentration_interval = 0.1 # 濃度間隔 0.1_10%
            type1 = 1  # Al(OH)3とラクトース
            type2 = 2  # ラクトースとマルトース
            type3 = 3  # マルトースとAl(OH)3
            if i == 0:
                self.concentration_color[0] = int(255)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(0)
            elif i == 1:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(255)
                self.concentration_color[2] = int(0)
            elif i == 2:
                self.concentration_color[0] = int(0)
                self.concentration_color[1] = int(0)
                self.concentration_color[2] = int(255)
            elif i == 3:
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
                    x, frequency_list = allread(self.inten_or_trans_or_reflect,i,j,self.last_type,self.last_num,self.from_frequency,
                                self.to_frequency,self.frequency_list).Frequency_trans_reflect_TDS_fre_trans_excel_cut_fre(self.file_list[j-1],self.feature_fre_list)
                    #print(self.file_list[j-1])
                    self.file_name_list.append(j)

                    if self.flag == 0:
                        x_all = x
                        self.flag += 1

                    else:
                        x_all = np.append(x_all, x, axis=0)


                    #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                    self.y_all.append(i)
                    self.y_all_dnn.append(i)
                except FileNotFoundError as e:
                    print(e)
        #print(concentration_color_type)
        return x_all, self.y_all, self.file_name_list, self.type_name_list, concentration_color_type, self.y_all_dnn, frequency_list


read = Read()

if cut_fre == 0:
    x_all, y_all, file_name_list, type_name_list, concentration_color_type, y_all_dnn, frequency_list, first_index = read.read()
elif cut_fre == 1:
    x_all, y_all, file_name_list, type_name_list, concentration_color_type, y_all_dnn, frequency_list = read.read_cut_fre()
    first_index = 12    #引数が足りないため、意味はない
else:
    print('select cut_fre with 0 or 1')

train_x,train_y,test_x,test_y = decide_test_number(x_all,y_all,test_number)
train_y = np.array(train_y)
test_y = np.array(test_y)

#referenceのカラーコード
#colorcode(test_y,width,length)
name = 'Reference'
colorcode_new(test_y, width, length, concentration_color_type, name)

#print(train_x)
#print(train_y)
'''
basemodel = KNeighborsClassifier()
basemodel.fit(train_x,train_y)
base_pred = basemodel.predict(test_x)
#print(base_pred)
'''
if feedbck == 0:

    #xgboost
    print('\nxgboost')
    name = 'xgboost'
    #train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
    #train_y_xgb = np.array(train_y_xgb)
    #test_y_xgb = np.array(test_y_xgb)
    #best_pred_1, xgb = xgboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index,class_number)
    best_pred_1, xgb = xgboost(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list, first_index,class_number)
    colorcode_new(best_pred_1, width, length, concentration_color_type, name)

    #Lightgbm
    print('\nLightGBM')
    name = 'LightGBM'
    #train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
    #train_y_xgb = np.array(train_y_xgb)
    #test_y_xgb = np.array(test_y_xgb)
    #best_pred, xgb = xboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index,class_number)
    #best_pred_2, lgbm =lightgbm(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index)
    best_pred_2, lgbm =lightgbm(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list, first_index)
    colorcode_new(best_pred_2, width, length, concentration_color_type, name)

    #CatBoost
    print('\nCatBoost')
    name = 'CatBoost'
    #train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
    #train_y_xgb = np.array(train_y_xgb)
    #test_y_xgb = np.array(test_y_xgb)
    #best_pred_3, cat = catboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index)
    best_pred_3, cat = catboost(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list, first_index)
    colorcode_new(best_pred_3, width, length, concentration_color_type, name)

    #SVM
    print('\nSVM')
    name = 'SVM'
    best_pred_4, sv=svm(train_x,train_y,test_x,test_y)
    #colorcode(best_pred,width,length)
    colorcode_new(best_pred_4, width, length, concentration_color_type, name)

    #k近傍法
    print('\nk近傍法')
    name = 'k_kinbou'
    best_pred_5,knn =kNN(train_x,train_y,test_x,test_y)
    #colorcode(best_pred,width,length)
    colorcode_new(best_pred_5, width, length, concentration_color_type, name)

    # PCA-SVM
    print('\nPCA-SVM')
    name = 'PCA-SVM'
    transformed, targets = pCA(x_all, y_all,pca_third_argument, file_name_list, type_name_list, concentration_color_type)
    train_x_pca,train_y_pca,test_x_pca,test_y_pca = decide_test_number(transformed,targets,test_number)
    best_pred_6, sv2 = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
    #colorcode(best_pred, width, length)
    colorcode_new(best_pred_6, width, length, concentration_color_type, name)

    #randomforest
    print('\nRandomfroest')
    name = 'Random_forest'
    best_pred_7, rf = randomforest(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list, first_index)
    #colorcode(best_pred,width,length)
    colorcode_new(best_pred_7, width, length, concentration_color_type, name)

    '''
    ## ここからDNN
    name = 'DNN'
    Y_ = np_utils.to_categorical(np.array(y_all_dnn), class_number)
    # train_x,train_y,test_x,test_y = train_test_split(x_all,Y_,1)#trainを選択する場合
    train_x_dnn, train_y_dnn, test_x_dnn, test_y_dnn = decide_test_number_onehot(x_all, Y_, test_number)  # testを選択する場合
    best_pred_8, probability = dnn_classification(train_x_dnn, train_y_dnn, test_x_dnn, test_y_dnn, class_number,
                                                  base_dir, from_frequency, to_frequency, frequency_list)
    print('\nDNN')
    print('best_pred:{}'.format(best_pred_8))
    print('probability:{}'.format(probability))
    print('average_probability:{}'.format(sum(probability) / len(probability)))
    colorcode_new(best_pred_8, width, length, concentration_color_type, name)
    '''

    #print(xgb)
    #print(lgbm)
    ####統合Stacking
    print('\nStacking_predict_with_all')
    name = 'Stacking_predict_with_all'
    estimators = [('xbg', xgb), ('lgbm', lgbm), ('cat', cat), ('sv', sv), ('knn', knn), ('rf', rf)]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=600), )
    clf.fit(train_x, train_y)
    y_pred = clf.predict_proba(test_x)
    # print(y_pred)
    best_pred = []
    probability = []
    for (i, pre) in enumerate(y_pred):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        # best_pred.append(category[y])
        best_pred.append(y)
        probability.append(pre[y])
    best_pred = np.array(best_pred)
    acc = clf.score(test_x, test_y)

    print('\nStacking_統合')
    print('Best score: {}'.format(acc))
    print('Best pred: {}'.format(best_pred))
    print('test_score: {}'.format(probability))
    colorcode_new(best_pred, width, length, concentration_color_type, name)


    # # 偏回帰係数、切片の取得
    # a = clf.coef_
    # a = np.abs(a)
    # b = clf.intercept_
    #
    # # グラフの作成
    # sns.set()
    # sns.set_style('whitegrid')
    # sns.set_palette('gray')
    #
    # x = np.array(
    #     ['RF_dep=2', 'RF_dep=4', 'RF_dep=6', 'RF_dep=8', 'RF_dep=10', 'LGBM_num=10', 'LGBM_num=100', 'LGBM_num=1000',
    #      'MLT'])
    # y = a
    #
    # x_position = np.arange(len(x))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.barh(x_position, y, tick_label=x)
    # ax.set_xlabel('Adoption rate')
    # ax.set_ylabel('Method')
    # fig.show()



else:
    pass

#####周波数選択用
if feedbck == 1:
    if wrapper_method == 0:
        # xgboost
        print('\nxgboost')
        name = 'xgboost'
        # train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
        # train_y_xgb = np.array(train_y_xgb)
        # test_y_xgb = np.array(test_y_xgb)
        # best_pred_1, xgb = xgboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index,class_number)
        best_pred_1, xgb = xgboost(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list,
                                   first_index, class_number)
        colorcode_new(best_pred_1, width, length, concentration_color_type, name)

        wrapper_method = xgb
        wrapper_name = 'Xgboost'

    elif wrapper_method == 1:
        # SVM
        print('\nSVM')
        name = 'SVM'
        best_pred_4, sv = svm(train_x, train_y, test_x, test_y)
        # colorcode(best_pred,width,length)
        colorcode_new(best_pred_4, width, length, concentration_color_type, name)

        wrapper_method = sv
        wrapper_name = 'SVM'

    elif wrapper_method == 2:
        # k近傍法
        print('\nk近傍法')
        name = 'k_kinbou'
        best_pred_5, knn = kNN(train_x, train_y, test_x, test_y)
        # colorcode(best_pred,width,length)
        colorcode_new(best_pred_5, width, length, concentration_color_type, name)

        wrapper_method = knn
        wrapper_name = 'KNN'

    elif wrapper_method == 3:
        # PCA-SVM
        print('\nPCA-SVM')
        name = 'PCA-SVM'
        transformed, targets = pCA(x_all, y_all, pca_third_argument, file_name_list, type_name_list,
                                   concentration_color_type)
        train_x_pca, train_y_pca, test_x_pca, test_y_pca = decide_test_number(transformed, targets, test_number)
        best_pred_6, sv2 = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
        # colorcode(best_pred, width, length)
        colorcode_new(best_pred_6, width, length, concentration_color_type, name)

        wrapper_method = sv2
        wrapper_name = 'PCA-SVM'

    elif wrapper_method == 4:
        # randomforest
        print('\nRandomforest')
        name = 'Random_forest'
        best_pred_7, rf = randomforest(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list,
                                       first_index)
        # colorcode(best_pred,width,length)
        colorcode_new(best_pred_7, width, length, concentration_color_type, name)

        # Boruta_and_random_forest
        print('\nBoruta_by_randomforest')
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
        feat_selector.fit(train_x, train_y)
        selected = feat_selector.support_
        print('selected_frequency_number: %d' % np.sum(selected))
        # print(selected)
        #numpyの特徴量抽出をコピーし、一つはlistに変換し出力、ndarrayはグラフ用
        boruta_fre_fea_np = frequency_list[selected]
        boruta_fre_fea_list = boruta_fre_fea_np.tolist()
        print('selected_frequency : {}'.format(boruta_fre_fea_list))


        #グラフ
        y = selected
        x = frequency_list
        sns.set()
        plt.bar(x, y, width=0.005, align="center")
        #plt.xticks([1.09,1.10,1.11,1.12,1.13,1.15])
        plt.title('Boruta Feature Importance', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Frequency[THz]', fontsize=20)
        plt.ylabel('feature importance', fontsize=20)
        plt.show()

        for a, b in enumerate(selected):
            if b == True:
                max_feature_list.append(first_index + a)
        print('selected_frequency_list : {}'.format(max_feature_list))

        wrapper_method = rf
        wrapper_name = 'Randomforest'

    elif wrapper_method == 5:
        # Lightgbm
        print('\nLightGBM')
        name = 'LightGBM'
        # train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
        # train_y_xgb = np.array(train_y_xgb)
        # test_y_xgb = np.array(test_y_xgb)
        # best_pred, xgb = xboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index,class_number)
        # best_pred_2, lgbm =lightgbm(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index)
        best_pred_2, lgbm = lightgbm(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list,
                                     first_index)
        colorcode_new(best_pred_2, width, length, concentration_color_type, name)

        wrapper_method = lgbm
        wrapper_name = 'LightGBM'

    elif wrapper_method == 6:
        # CatBoost
        print('\nCatBoost')
        name = 'CatBoost'
        # train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb = decide_test_number(x_all,y_all_dnn,test_number)
        # train_y_xgb = np.array(train_y_xgb)
        # test_y_xgb = np.array(test_y_xgb)
        # best_pred_3, cat = catboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index)
        best_pred_3, cat = catboost(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list,
                                    first_index)
        colorcode_new(best_pred_3, width, length, concentration_color_type, name)

        wrapper_method = cat
        wrapper_name = 'CatBoost'

    else:
        print('select wrapper method')

    # ラッパー法 stepForwardselection
    print('\nラッパー法(step_forward_selection) : {}'.format(wrapper_name))
    sfs = sFs(wrapper_method,train_x,train_y,frequency_list,first_index)

    # ラッパー法 stepBackwardselection
    print('\nラッパー法(step_backward_selection) : {}'.format(wrapper_name))
    sbs = sBs(wrapper_method, train_x, train_y, frequency_list, first_index)

else:
    pass


'''
#ラッパー法 stepForwardselection
print('\nラッパー法(step_forward_selection) : {}'.format(wrapper_name))
sfs = SFS(wrapper_method, k_features=10, forward=True, floating=False, verbose=2, scoring='accuracy', cv=3)
sfs = sfs.fit(train_x,train_y)
tuple_selected = sfs.k_feature_idx_
selected = list(tuple_selected)
print('CV_score:{}'.format(sfs.k_score_))
print('seleted_frequency_number:{}'.format(len(selected)))
print('selected_frrequency:{}'.format(frequency_list[selected]))

for a,b in enumerate(selected):
        max_feature_list.append(first_index + b)
print('selected_frequency_list:{}'.format(max_feature_list))


#ラッパー法 stepBackwardselection
print('\nラッパー法(step_backward_selection) : {}'.format(wrapper_name))
sbs = SFS(wrapper_method, k_features=10, forward=False, floating=False, verbose=2, scoring='accuracy', cv=3)
sbs = sbs.fit(train_x,train_y)
tuple_selected = sbs.k_feature_idx_
selected = list(tuple_selected)
print('CV_score:{}'.format(sbs.k_score_))
print('selected_frequency_number:{}'.format(len(selected)))
print('selected_frrequency:{}'.format(frequency_list[selected]))

for a,b in enumerate(selected):
        max_feature_list.append(first_index + b)
print('selected_frequency_list:{}'.format(max_feature_list))


#Boruta_and_random_forest
print('\nBoruta_by_randomforest')
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(train_x, train_y)
selected = feat_selector.support_
print('selected_frequency_number: %d' %np.sum(selected))
#print(selected)
print('selected_frrequency : {}'.format(frequency_list[selected]))

for a,b in enumerate(selected):
    if b == True:
        max_feature_list.append(first_index + a)
print('selected_frequency_list : {}'.format(max_feature_list))
'''
'''

## ここからDNN
name = 'DNN'
Y_ = np_utils.to_categorical(np.array(y_all_dnn), class_number)
#train_x,train_y,test_x,test_y = train_test_split(x_all,Y_,1)#trainを選択する場合
train_x_dnn,train_y_dnn,test_x_dnn,test_y_dnn = decide_test_number_onehot(x_all,Y_,test_number)#testを選択する場合
best_pred_8, probability = dnn_classification(train_x_dnn, train_y_dnn, test_x_dnn, test_y_dnn, class_number, base_dir, from_frequency, to_frequency, frequency_list)
print('\nDNN')
print('best_pred:{}'.format(best_pred_8))
print('probability:{}'.format(probability))
print('average_probability:{}'.format(sum(probability)/len(probability)))
colorcode_new(best_pred_8, width, length, concentration_color_type, name)

####統合Stacking
print('\nStacking_predict_with_all')
name = 'Stacking_predict_with_all'
estimators = [('xbg', xgb),('lgbm', lgbm),('cat', cat),('sv', sv),('knn', knn),('rf', rf)]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=600),)
clf.fit(train_x,train_y)
y_pred = clf.predict_proba(test_x)
# print(y_pred)
best_pred = []
probability = []
for (i, pre) in enumerate(y_pred):
    y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
    # best_pred.append(category[y])
    best_pred.append(y)
    probability.append(pre[y])
best_pred = np.array(best_pred)
acc = clf.score(test_x,test_y)
print('\nStacking_統合')
print('Best score: {}'.format(acc))
print('Best pred: {}'.format(best_pred))
print('test_score: {}'.format(probability))
colorcode_new(best_pred, width, length, concentration_color_type, name)
'''


'''

estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svr', make_pipeline(
                StandardScaler(),
                LinearSVC(max_iter=4000, random_state=42)
                )
            )
        ]
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=600),
)
clf.fit(train_x, train_y)

print("正解率:",clf.score(test_x,test_y))


stacked_predictions = np.column_stack((best_pred_1,best_pred_2,best_pred_3,best_pred_4,best_pred_5,best_pred_6,best_pred_7,best_pred_8))

meta_model = LogisticRegression()
meta_model.fit(stacked_predictions, test_y)
valid_pred_1 = xgb.fit()
'''