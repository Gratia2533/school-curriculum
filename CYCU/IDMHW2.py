# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:16:12 2022

@author: USER
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#%%train data
train = pd.read_csv("pokemon_train.csv")#讀取資料集
#train.set_index("ID",inplace=True)#將ID設為index
X_train = train.drop(columns=['Type.1','Type.2','ID'])#去除X不需要的資料
y_train = train['Type.1']#設定要預測的目標
#%%test data
test = pd.read_csv("pokemon_test.csv")
#test.set_index("ID",inplace=True)
X_test = test.drop(columns=['Type.1','Type.2','ID'])
y_test = test['Type.1']

#%%valid data
valid = pd.read_csv("pokemon_valid.csv")
#valid.set_index("ID",inplace=True)
X_valid = valid.drop(columns=['Type.1','Type.2','ID'])
y_valid = valid['Type.1']
#%%KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_y_predict = knn_model.predict(X_valid)
accuracy_score(y_valid, knn_y_predict)

#%%DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(random_state=53)#建立決策樹模型
dtree_model.fit(X_train, y_train)#用trainData去訓練
dtree_y_predict = dtree_model.predict(X_valid)#預測驗證集
accuracy_score(y_valid, dtree_y_predict)
#%%RandomForestClassifier
rfc_model = RandomForestClassifier(random_state=53)
rfc_model.fit(X_train, y_train)
rfc_y_predict = rfc_model.predict(X_valid)
accuracy_score(y_valid, rfc_y_predict)
#%%Use RandomForest with test Set
#accuracy_score(y_test, knn_y_predict)
#accuracy_score(y_test, dtree_y_predict)
accuracy_score(y_test, rfc_y_predict)#使用RandomForest的模型去預測testData