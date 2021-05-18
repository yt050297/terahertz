from sklearn.svm import SVC,SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB # ガウシアン
import datetime
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers.convolutional import Conv1D, UpSampling1D
import keras.backend.tensorflow_backend as KTF
from keras import regularizers
import os
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import seaborn as sns

def gaussiannb(train_x, train_y, test_x, test_y):
    gnb = GaussianNB()
    clf = gnb.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    score = accuracy_score(y_pred, test_y)
    print('score: {}'.format(score))

    return y_pred

def xgboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index,class_number):
    #param_grid = {"max_depth": range(3,10,1),'min_child_weight':range(1,6,1)}
    param_grid = {}
    xgb_grid = GridSearchCV(estimator=XGBClassifier(booster="gbtree", silent=True, verbosity=0, objective='multi:softprob'),
                       param_grid=param_grid, cv=3, verbose=False, n_jobs=5)
    #verbose=Trueで実行様子を表示
    '''
    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(booster="gbtree", silent=1, verbosity=0, objective='multi:softprob' ),
    
        param_grid={
            "max_depth":[2,3,6,10],
            "subsample":[0.5,0.8,0.9,1],
            "colsample_bytree":[0.5,1],
            "learning_rate":[0.001,0.01,0.1,0.3],
            "eta":[0.3,0.15,0.10]
        },
        cv=3,n_jobs=5)
    '''

    xgb_grid.fit(train_x_xgb, train_y_xgb)
    xgb_grid_best = xgb_grid.best_estimator_
    y_pred = xgb_grid_best.predict_proba(test_x_xgb)
    #print(y_pred)
    best_pred = []
    probability = []
    # category = np.arange(1, class_number + 1)
    for (i, pre) in enumerate(y_pred):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        # best_pred.append(category[y])
        best_pred.append(y)
        probability.append(pre[y])
    best_pred = np.array(best_pred)
    # Accuracyの計算
    acc = accuracy_score(test_y_xgb, best_pred)
    print('\nxgboost')
    print("Best Model Parameter: ", xgb_grid.best_params_)
    print('Best score: {}'.format(acc))
    print('Best pred: {}'.format(best_pred))
    print('test_score: {}'.format(probability))

    # feature importanceを表示
    plt.figure(figsize=(10, 5))
    # xgb.plot_importance(bst)
    feature_importances = xgb_grid.best_estimator_.feature_importances_
    y = feature_importances
    x = frequency_list

    ###特徴量の多い5個の周波数を表示
    max_feature = y.argsort()[::-1]
    max_feature2 = max_feature + first_index
    # max_feature_tolist = max_feature.tolist()
    max_feature_tolist2 = max_feature2.tolist()
    print(max_feature_tolist2)

    # print(x[max_feature[0]])
    # print(x[max_feature[0]])
    # print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}'.format(x[max_feature[0]], x[max_feature[1]],
    #                                                                x[max_feature[2]], x[max_feature[3]],
    #                                                                x[max_feature[4]], x[max_feature[5]],
    #                                                                x[max_feature[6]], x[max_feature[7]],
    #                                                                x[max_feature[8]], x[max_feature[9]],x[max_feature[10]], x[max_feature[11]],x[max_feature[12]], x[max_feature[13]],x[max_feature[14]], x[max_feature[15]],x[max_feature[16]], x[max_feature[17]],x[max_feature[18]], x[max_feature[19]],x[max_feature[20]], x[max_feature[21]],x[max_feature[22]], x[max_feature[23]],x[max_feature[24]], x[max_feature[25]],x[max_feature[26]], x[max_feature[27]]))
    # sns.set()
    # plt.bar(x, y, width=0.005, align="center")
    # plt.title('XGBoost Feature Importance', fontsize=20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel('Frequency[THz]', fontsize=20)
    # plt.ylabel('feature importance', fontsize=20)
    # plt.show()

    best_pred_1_start = best_pred
    xgb = xgb_grid.best_estimator_

    return best_pred_1_start, xgb

'''
    ##細かい用
    dtrain = xgb.DMatrix(train_x_xgb, label=train_y_xgb)
    dtest = xgb.DMatrix(test_x_xgb, label=test_y_xgb)
    param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob', 'num_class': class_number}
    num_round = 100
    evals = [(dtrain, 'train'), (dtest, 'eval')]  # 学習に用いる検証用データ
    evaluation_results = {}  # 学習の経過を保存する箱
    # 表示したいとき
    # bst = xgb.train(param, dtrain, num_round, evals=evals, evals_result=evaluation_results, verbose_eval=1)
    bst = xgb.train(param, dtrain, num_round)

    # テストデータで予測
    y_pred = bst.predict(dtest)

    best_pred = []
    probability = []
    # category = np.arange(1, class_number + 1)
    for (i, pre) in enumerate(y_pred):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        # best_pred.append(category[y])
        best_pred.append(y)
        probability.append(pre[y])
    best_pred = np.array(best_pred)
    # Accuracyの計算
    acc = accuracy_score(test_y_xgb, best_pred)
    print('Best score: {}'.format(acc))
    print('Best pred: {}'.format(best_pred))
    print('test_score: {}'.format(probability))

    # 学習過程の可視化
    plt.plot(evaluation_results['train']['mlogloss'], label='train')
    plt.plot(evaluation_results['eval']['mlogloss'], label='eval')
    plt.ylabel('Log loss')
    plt.xlabel('Boosting round')
    plt.title('Training performance')
    plt.legend()
    plt.show()


    #xgb.plot_importance(bst)
    '''


def lightgbm(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index):

    #param_grid = {"max_depth":[10,25,50,75],"num_leaves":[100,300,900,1200]}
    param_grid = {}
    '''
    param_grid = {"max_depth": [10, 25, 50, 75],
                  "learning_rate": [0.001, 0.01, 0.05, 0.1],
                  "num_leaves": [100, 300, 900, 1200],
                  "n_estimators": [100, 200, 500]
                  }
    '''
    lgbm_grid = GridSearchCV(estimator = LGBMClassifier(silent=True,objective="multi_logloss"),
                             param_grid = param_grid, scoring = 'balanced_accuracy',
                             cv =3, verbose=False, return_train_score = True, n_jobs=-1)
    # verbose=Trueで実行様子を表示

    lgbm_grid.fit(train_x_xgb,train_y_xgb)
    lgbm_grid_best = lgbm_grid.best_estimator_
    y_pred = lgbm_grid_best.predict_proba(test_x_xgb)
    #print(y_pred)

    best_pred = []
    probability = []
    # category = np.arange(1, class_number + 1)
    for (i, pre) in enumerate(y_pred):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        # best_pred.append(category[y])
        best_pred.append(y)
        probability.append(pre[y])
    best_pred = np.array(best_pred)
    # Accuracyの計算
    acc = accuracy_score(test_y_xgb, best_pred)
    print('\nLightGBM')
    print("Best Model Parameter: ", lgbm_grid.best_params_)
    print('Best score: {}'.format(acc))
    print('Best pred: {}'.format(best_pred))
    print('test_score: {}'.format(probability))

    # feature importanceを表示
    plt.figure(figsize=(10, 5))
    # xgb.plot_importance(bst)
    feature_importances = lgbm_grid.best_estimator_.feature_importances_
    y = feature_importances
    x = frequency_list

    ###特徴量の多い5個の周波数を表示
    max_feature = y.argsort()[::-1]
    max_feature2 = max_feature + first_index
    # max_feature_tolist = max_feature.tolist()
    max_feature_tolist2 = max_feature2.tolist()
    print(max_feature_tolist2)

    # print(x[max_feature[0]])
    # print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}'.format(x[max_feature[0]], x[max_feature[1]],
    #                                                                x[max_feature[2]], x[max_feature[3]],
    #                                                                x[max_feature[4]], x[max_feature[5]],
    #                                                                x[max_feature[6]], x[max_feature[7]],
    #                                                                x[max_feature[8]], x[max_feature[9]]))
    print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}'.format(x[max_feature[0]], x[max_feature[1]],
                                                                   x[max_feature[2]], x[max_feature[3]],
                                                                   x[max_feature[4]], x[max_feature[5]],
                                                                   x[max_feature[6]], x[max_feature[7]],
                                                                   x[max_feature[8]], x[max_feature[9]],x[max_feature[10]], x[max_feature[11]],x[max_feature[12]], x[max_feature[13]],x[max_feature[14]], x[max_feature[15]],x[max_feature[16]], x[max_feature[17]],x[max_feature[18]], x[max_feature[19]],x[max_feature[20]], x[max_feature[21]],x[max_feature[22]], x[max_feature[23]],x[max_feature[24]], x[max_feature[25]],x[max_feature[26]], x[max_feature[27]]))
    sns.set()
    plt.bar(x, y, width=0.005, align="center")
    plt.title('LightGBM Feature Importance', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Frequency[THz]', fontsize=20)
    plt.ylabel('feature importance', fontsize=20)
    plt.show()

    best_pred_1_start = best_pred
    lgbm = lgbm_grid.best_estimator_

    return best_pred_1_start, lgbm

def catboost(train_x_xgb,train_y_xgb,test_x_xgb,test_y_xgb,from_frequency,to_frequency,frequency_list, first_index):

    #param_grid = {"max_depth":[10,25,50,75],"num_leaves":[100,300,900,1200]}
    param_grid = {}
    '''
    param_grid = {"max_depth": [10, 25, 50, 75],
                  "learning_rate": [0.001, 0.01, 0.05, 0.1],
                  "num_leaves": [100, 300, 900, 1200],
                  "n_estimators": [100, 200, 500]
                  }
    '''
    cat_grid = GridSearchCV(estimator = CatBoostClassifier(iterations=500,
                                                           silent=True),
                             param_grid = param_grid, scoring = 'balanced_accuracy',
                             cv =3, verbose=False, return_train_score = True, n_jobs=-1)
    cat_grid.fit(train_x_xgb,train_y_xgb)
    cat_grid_best =cat_grid.best_estimator_
    y_pred = cat_grid_best.predict_proba(test_x_xgb)
    #print(y_pred)

    best_pred = []
    probability = []
    # category = np.arange(1, class_number + 1)
    for (i, pre) in enumerate(y_pred):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        # best_pred.append(category[y])
        best_pred.append(y)
        probability.append(pre[y])
    best_pred = np.array(best_pred)
    # Accuracyの計算
    acc = accuracy_score(test_y_xgb, best_pred)
    print('\nCatBoost')
    print("Best Model Parameter: ", cat_grid.best_params_)
    print('Best score: {}'.format(acc))
    print('Best pred: {}'.format(best_pred))
    print('test_score: {}'.format(probability))

    # feature importanceを表示
    plt.figure(figsize=(10, 5))
    # xgb.plot_importance(bst)
    feature_importances = cat_grid.best_estimator_.feature_importances_
    y = feature_importances
    x = frequency_list

    ###特徴量の多い5個の周波数を表示
    max_feature = y.argsort()[::-1]
    max_feature2 = max_feature + first_index
    # max_feature_tolist = max_feature.tolist()
    max_feature_tolist2 = max_feature2.tolist()
    print(max_feature_tolist2)

    # print(x[max_feature[0]])
    # print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}'.format(x[max_feature[0]], x[max_feature[1]],
    #                                                                x[max_feature[2]], x[max_feature[3]],
    #                                                                x[max_feature[4]], x[max_feature[5]],
    #                                                                x[max_feature[6]], x[max_feature[7]],
    #                                                                x[max_feature[8]], x[max_feature[9]]))
    print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}'.format(x[max_feature[0]], x[max_feature[1]],
                                                                   x[max_feature[2]], x[max_feature[3]],
                                                                   x[max_feature[4]], x[max_feature[5]],
                                                                   x[max_feature[6]], x[max_feature[7]],
                                                                   x[max_feature[8]], x[max_feature[9]],x[max_feature[10]], x[max_feature[11]],x[max_feature[12]], x[max_feature[13]],x[max_feature[14]], x[max_feature[15]],x[max_feature[16]], x[max_feature[17]],x[max_feature[18]], x[max_feature[19]],x[max_feature[20]], x[max_feature[21]],x[max_feature[22]], x[max_feature[23]],x[max_feature[24]], x[max_feature[25]],x[max_feature[26]], x[max_feature[27]]))
    sns.set()
    plt.bar(x, y, width=0.005, align="center")
    plt.title('CatBoost Feature Importance', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Frequency[THz]', fontsize=20)
    plt.ylabel('feature importance', fontsize=20)
    plt.show()

    best_pred_1_start = best_pred
    cat = cat_grid.best_estimator_

    return best_pred_1_start, cat

def randomforest(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list, first_index):
    # use a full grid over all parameters
    param_grid = {"n_estimators": np.arange(50,300,10)}
    #param_grid = {}

    forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0,bootstrap = True),
                               param_grid=param_grid,
                               cv=3)
    forest_grid.fit(train_x, train_y)  # fit
    forest_grid_best = forest_grid.best_estimator_  # best estimator
    best_pred = forest_grid_best.predict(test_x)
    best_score = accuracy_score(best_pred, test_y)
    print("Best Model Parameter: ", forest_grid.best_params_)
    print('Best score: {}'.format(best_score))
    print('Best pred:{}'.format(best_pred))

    # 特徴量の重要度
    feature_importances = forest_grid_best.feature_importances_
    y = feature_importances
    x = frequency_list

    '''
    if not frequency_list:
        x = np.arange(from_frequency,to_frequency,0.01)
    else:
        x = frequency_list
    '''
    ###特徴量の多い5個の周波数を表示
    max_feature = y.argsort()[::-1]
    max_feature2 = max_feature + first_index
    #max_feature_tolist = max_feature.tolist()
    max_feature_tolist2 = max_feature2.tolist()
    print(max_feature_tolist2)

    #print(x[max_feature[0]])
    # print(x[max_feature[0]])
    # print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9} '.format(x[max_feature[0]], x[max_feature[1]],
    #                                                                x[max_feature[2]], x[max_feature[3]],
    #                                                                x[max_feature[4]], x[max_feature[5]],
    #                                                                x[max_feature[6]], x[max_feature[7]],
    #                                                                x[max_feature[8]], x[max_feature[9]]))
    print('Most_feature_frequency: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}'.format(x[max_feature[0]], x[max_feature[1]],
                                                                   x[max_feature[2]], x[max_feature[3]],
                                                                   x[max_feature[4]], x[max_feature[5]],
                                                                   x[max_feature[6]], x[max_feature[7]],
                                                                   x[max_feature[8]], x[max_feature[9]],x[max_feature[10]], x[max_feature[11]],x[max_feature[12]], x[max_feature[13]],x[max_feature[14]], x[max_feature[15]],x[max_feature[16]], x[max_feature[17]],x[max_feature[18]], x[max_feature[19]],x[max_feature[20]], x[max_feature[21]],x[max_feature[22]], x[max_feature[23]],x[max_feature[24]], x[max_feature[25]],x[max_feature[26]], x[max_feature[27]]))
    #print(x)
    #print(y)
    #print(len(x))
    #print(len(y))
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, width = 0.005, align="center")
    plt.title('RandomForest Feature Importance', fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.bar(x, y)
    plt.xlabel('Frequency[THz]', fontsize=20)
    plt.ylabel('feature importance', fontsize=20)
    plt.show()

    rf = forest_grid.best_estimator_

    return best_pred, rf


def svm(train_x, train_y, test_x, test_y):
    param_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]
    #param_list = [0.01,0.1, 1,10]
    best_score = 0
    best_parameters = {}
    kernel = 'rbf'
    for gamma in param_list:  # グリッドサーチをしてハイパーパラメータ探索
        for C in param_list:
            estimator = SVC(gamma=gamma, kernel=kernel, C=C)
            classifier = OneVsRestClassifier(estimator)
            classifier.fit(train_x, train_y)
            pred_y = classifier.predict(test_x)
            classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
            classifier2.fit(train_x, train_y)
            pred_y2 = classifier2.predict(test_x)
            onerest_score = accuracy_score(test_y, pred_y)
            oneone_score = accuracy_score(test_y, pred_y2)
            if onerest_score > oneone_score:
                score = onerest_score
                hikaku = 'One-versus-the-rest'
                better_pred = pred_y
            else:
                score = oneone_score
                hikaku = 'One-versus-one'
                better_pred = pred_y2
            # 最も良いスコアのパラメータとスコアを更新
            if score > best_score:
                best_hikaku = hikaku
                best_score = score
                best_parameters = {'gamma': gamma, 'C': C}
                best_pred = better_pred

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))
    print('比較方法:{}'.format(best_hikaku))
    print('Best pred:{}'.format(best_pred))

    sv = classifier2
    return best_pred,sv

def svm_gridsearchcv(train_x, train_y, test_x, test_y):
    ### 探索するパラメータ空間
    def param():
        ret = {
            'C': np.linspace(0.0001, 1000, 10),
            'kernel': ['rbf', 'linear', 'poly'],
            'degree': np.arange(1, 6, 1),
            'gamma': np.linspace(0.0001, 1000, 10)
        }
        return ret
    # GridSearchCVのインスタンスを作成&学習&スコア記録
    gscv_one = GridSearchCV(SVC(), param(), cv=3,return_train_score=False, verbose=0)
    gscv_one.fit(train_x, train_y)
    # 最高性能のモデルを取得
    best_one_vs_one = gscv_one.best_estimator_
    best_pred_one = best_one_vs_one.predict(test_x)
    oneone_score = accuracy_score(test_y, best_pred_one)
    parameters_one =gscv_one.best_params_
    ##one_versus_the_rest
    classifier = OneVsRestClassifier(estimator=SVC())
    parameters = {
        'estimator__C': np.linspace(0.0001, 1000, 10),
        'estimator__kernel': ['rbf', 'linear', 'poly'],
        'estimator__degree': np.arange(1, 6, 1),
        'estimator__gamma': np.linspace(0.0001, 1000, 10)
    }
    gscv_rest = GridSearchCV(estimator = classifier,param_grid = parameters,cv=3,return_train_score=False, verbose=0)
    gscv_rest.fit(train_x, train_y)
    # 最高性能のモデルを取得
    best_one_vs_rest = gscv_rest.best_estimator_
    best_pred_rest = best_one_vs_rest.predict(test_x)
    onerest_score = accuracy_score(test_y, best_pred_rest)
    parameters_rest =gscv_rest.best_params_

    if oneone_score > onerest_score:
        best_score = oneone_score
        best_compare = 'One-versus-one'
        best_pred = best_pred_one
        best_parameters = parameters_one
    else:
        best_score = onerest_score
        best_compare = 'One-versus-the-rest'
        best_pred = best_pred_rest
        best_parameters = parameters_rest

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))
    print('比較方法:{}'.format(best_compare))
    print('Best pred:{}'.format(best_pred))
    # 混同行列を出力
    #print(confusion_matrix(test_y, best_pred))
    return best_pred




def kNN(train_x, train_y, test_x, test_y):
    k_list = [1,2,3]  # k の数
    weights_list = ['uniform', 'distance']
    ac_score_compare = 0
    for weights in weights_list:
        for k in k_list:
            clf = neighbors.KNeighborsClassifier(k, weights=weights)
            clf.fit(train_x, train_y)
            # 正答率を求める
            pred_y = clf.predict(test_x)
            ac_score = metrics.accuracy_score(pred_y, test_y)
            # print(type(k))
            # print(type(iris_y_test))


            if ac_score_compare == 0:
                ac_score_compare = ac_score
                best_pred = pred_y
                best_k = k
                best_weight = weights
                best_accuracy = ac_score
            elif ac_score_compare < ac_score:
                best_pred = pred_y
                best_k = k
                best_weight = weights
                best_accuracy = ac_score
    print('k={0},weight={1}'.format(best_k, best_weight))
    print('正答率 =', best_accuracy)
    print(pred_y)
    knn = clf
    return pred_y, knn

def sFs(wrapper_method,train_x,train_y,frequency_list,first_index):
    max_feature_list = []
    sfs = SFS(wrapper_method, k_features=15, forward=True, floating=False, verbose=2, scoring='accuracy', cv=3)
    sfs = sfs.fit(train_x,train_y)
    tuple_selected = sfs.k_feature_idx_
    selected = list(tuple_selected)
    print('CV_score:{}'.format(sfs.k_score_))
    print('selected_frequency_number:{}'.format(len(selected)))

    # numpyの特徴量抽出をコピーし、一つはlistに変換し出力、ndarrayはグラフ用
    SFS_fre_fea_np = frequency_list[selected]
    SFS_fre_fea_list = SFS_fre_fea_np.tolist()
    print('selected_frequency : {}'.format(SFS_fre_fea_list))

    #print('selected_frequency:{}'.format(frequency_list[selected]))

    for a,b in enumerate(selected):
            max_feature_list.append(first_index + b)
    print('selected_frequency_list:{}'.format(max_feature_list))


def sBs(wrapper_method, train_x, train_y, frequency_list, first_index):
    max_feature_list = []
    sfs = SFS(wrapper_method, k_features=15, forward=False, floating=False, verbose=2, scoring='accuracy', cv=3)
    sfs = sfs.fit(train_x, train_y)
    tuple_selected = sfs.k_feature_idx_
    selected = list(tuple_selected)
    print('CV_score:{}'.format(sfs.k_score_))
    print('selected_frequency_number:{}'.format(len(selected)))

    # numpyの特徴量抽出をコピーし、一つはlistに変換し出力、ndarrayはグラフ用
    SBS_fre_fea_np = frequency_list[selected]
    SBS_fre_fea_list = SBS_fre_fea_np.tolist()
    print('selected_frequency : {}'.format(SBS_fre_fea_list))
    #print('selected_frequency:{}'.format(frequency_list[selected]))

    for a, b in enumerate(selected):
        max_feature_list.append(first_index + b)
    print('selected_frequency_list:{}'.format(max_feature_list))


def pCA(x_all, y_all, number, file_name_list, type_name_list, concentration_color_type):
    features = x_all
    targets = y_all
    pca = PCA(n_components=2)
    pca.fit(features)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)
    # 主成分の寄与率を出力する
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
    '''
    # 主成分をプロットする
    if number == 0: #厚みの場合

        for label in np.unique(targets): #厚さのみのPCA
            plt.scatter(transformed[targets == label, 0],
                        transformed[targets == label, 1], label='{}mm'.format(label*0.5))
        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        plt.legend(loc= 'best',fontsize=16)
        #plt.yticks([-1.0,-0.5,0.0,0.5,1.0])
        plt.tick_params(labelsize=24)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)): #ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='red')
            elif item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='#%02X%02X%02X' % (0,255,0))

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='blue')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='#%02X%02X%02X' % (255,215,0))

        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        #plt.legend(loc='best',fontsize=16)
        #plt.xticks([-10, -5, 0, 5, 10])
        #plt.yticks([-10, -5, 0, 5, 10])
        plt.tick_params(labelsize=24)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)): #ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='red')
            elif item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='#%02X%02X%02X' % (0,255,0))

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='blue')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='#%02X%02X%02X' % (255,215,0))

        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        #plt.legend(loc='best',fontsize=16)
        #plt.xticks([-10, -5, 0, 5, 10])
        #plt.yticks([-10, -5, 0, 5, 10])
        plt.tick_params(labelsize=24)
        plt.show()

    else: #試薬の場合

        for label, name, concentration_color in zip(np.unique(targets), type_name_list, concentration_color_type):
            plt.scatter(transformed[targets == label, 0],
                        transformed[targets == label, 1], label=name, c = '#%02X%02X%02X' % (concentration_color[0],concentration_color[1],concentration_color[2]))

        plt.xlabel('pc1', fontsize=28)
        plt.ylabel('pc2', fontsize=28)
        #plt.xticks([-2, -1, -0, 0, 1, 2])
        #plt.yticks([-0.75, -0.5, -0.25, 0.00, 0.25, 0.5, 0.75])
        #plt.subplots_adjust(left=0.1, right=0.4, bottom=0.2, top=0.95)
        #plt.legend(loc='best', borderaxespad=0,bbox_to_anchor=(1.05, 1),fontsize=10,ncol=1)
        plt.tick_params(labelsize=24)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)):  # ファイル名も表記する。
            plt.scatter(transformed[index, 0],
                        transformed[index, 1], marker="${}$".format(file_name), c='red')

        plt.xlabel('pc1', fontsize=28)
        plt.ylabel('pc2', fontsize=28)
        #plt.xticks([-2, -1, -0, 0, 1, 2])
        #plt.yticks([-0.75, -0.5, -0.25, 0.00, 0.25, 0.5, 0.75])
        #plt.legend(loc='best', fontsize=16)
        plt.tick_params(labelsize=24)
        plt.show()
    '''
    targets_np = np.array(targets)

    return transformed, targets_np

def iCA(x_all, y_all):
    # 独立成分の数＝24
    decomposer = FastICA(n_components=2)
    # データの平均を計算
    M = np.mean(x_all, axis=1)[:, np.newaxis]
    # 各データから平均を引く
    data2 = x_all - M
    # 平均0としたデータに対して、独立成分分析を実施
    decomposer.fit(data2)

    # 独立成分ベクトルを取得(D次元 x 独立成分数)
    S = decomposer.transform(data2)
    #プロットする
    for label in np.unique(y_all):
        plt.scatter(S[y_all == label, 0],
                    S[y_all == label, 1], )
    plt.legend(loc='upper right',
               bbox_to_anchor=(1,1),
               borderaxespad=0.5,fontsize = 10)
    plt.title('principal component')
    plt.xlabel('Ic1')
    plt.ylabel('Ic2')

    # 主成分の寄与率を出力する
    #print('各次元の寄与率: {0}'.format(decomposer.explained_variance_ratio_))
    #print('累積寄与率: {0}'.format(sum(decomposer.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()


    return S, y_all

def smirnov_grubbs(data, alpha):
    x, o = list(data), []
    while True:
        n = len(x)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_min, i_max = np.argmin(x), np.argmax(x)
        myu, std = np.mean(x), np.std(x, ddof=1)
        i_far = i_max if np.abs(x[i_max] - myu) > np.abs(x[i_min] - myu) else i_min
        tau_far = np.abs((x[i_far] - myu) / std)
        if tau_far < tau: break
        o.append(x.pop(i_far))
    return np.array(x), np.array(o)


def dnn_classification(train_x, train_y, test_x, test_y, class_number, base_dir, from_frequency, to_frequency, frequency_list):
    # conv1 = 30
    nb_epoch = 1000
    nb_batch = 32
    learning_rate = 1e-2
    try:  ##convolutionを使う場合
        conv1

        train_x.resize(train_x.shape[0], train_x.shape[1], 1)
        test_x.resize(test_x.shape[0], test_x.shape[1], 1)
    except:
        pass

    dense1 = 60
    dense2 = 30
    dense3 = 14
    dense4 = class_number
    regularizers_l2_1 = 0
    regularizers_l2_2 = 0
    regularizers_l2_3 = 0

    try:
        model_structure = 'conv{0}relu_{1}relul2{2}_{3}relul2{4}_{5}relul2{6}_{7}softmax'.format(conv1, dense1,
                                                                                                 regularizers_l2_1,
                                                                                                 dense2,
                                                                                                 regularizers_l2_2,
                                                                                                 dense3,
                                                                                                 regularizers_l2_3,
                                                                                                 dense4)
    except:
        model_structure = '{0}relul2{1}_{2}relul2{3}_{4}relul2{5}_{6}softmax'.format(dense1, regularizers_l2_1, dense2,
                                                                                     regularizers_l2_2, dense3,
                                                                                     regularizers_l2_3, dense4)
    f_log = base_dir + '/logs/fit' + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    # print(f_log)
    f_model = base_dir + '/model'  + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    os.makedirs(f_model, exist_ok=True)
    # ニュートラルネットワークで使用するモデル作成
    old_session = KTF.get_session()
    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        model = keras.models.Sequential()
        try:
            model.add(Conv1D(conv1, 4, padding='same', input_shape=(train_x.shape[1:]), activation='relu'))
            model.add(Flatten())
            model.add(Dense(dense1, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_1)))
        except:
            model.add(Dense(dense1, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_1),
                            input_shape=(train_x.shape[1:])))

        # model.add(Dropout(0.25))
        model.add(Dense(dense2, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_2)))
        # model.add(Dropout(0.25))
        model.add(Dense(dense3, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_3)))
        model.add(Dense(dense4, activation='softmax'))

        model.summary()
        # optimizer には adam を指定
        adam = keras.optimizers.Adam(lr=learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        train_y = np.array(train_y)
        test_y = np.array(test_y)
        # print(test_y)
        # print(test_y.shape)
        # print(type(test_y))
        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=0, mode='auto')
        tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
        # cp_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(f_model,'tag_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        # cbks = [es_cb, tb_cb, cp_cb]
        cbks = [es_cb, tb_cb]
        history = model.fit(train_x, train_y, batch_size=nb_batch, epochs=nb_epoch,
                            validation_data=(test_x, test_y), callbacks=cbks, verbose=1)
        score = model.evaluate(test_x, test_y, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        predict = model.predict(test_x)
        # print('predict:{}'.format(predict))
        print('save the architecture of a model')
        json_string = model.to_json()
        open(os.path.join(f_model, 'tag_model.json'), 'w').write(json_string)
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, 'tag_model.yaml'), 'w').write(yaml_string)
        print('save weights')
        model.save_weights(os.path.join(f_model, 'tag_weights.hdf5'))
    KTF.set_session(old_session)
    best_pred = []
    probability = []
    category = np.arange(0, class_number)
    #1スタート
    #category = np.arange(1, class_number+1)
    for (i, pre) in enumerate(predict):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        best_pred.append(category[y])
        probability.append(pre[y])
    return best_pred, probability
