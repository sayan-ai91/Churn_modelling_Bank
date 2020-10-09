# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:43:06 2020

@author: Sayan Mondal
"""

## Churn_MODELLING prediction

import pandas as pd
import numpy as np
import sweetviz ## For EDA just by a click...##
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt


data1=pd.read_csv("churn_Modelling.csv")

data1.describe()
data1.columns

## droping the unnecessary columns which dont have impact on output..##
data=data1.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

data.head()

data.isna().sum() ## no missing values..##


data2=data.corr()
sns.heatmap(data2,cmap='RdYlGn') 

de_report=sweetviz.analyze([data, "data"],target_feat="Exited")
de_report.show_html('report.html') ## this will create a detailed html file of all plots as well as analysis


# Creating Dummies..
data = pd.get_dummies(data, columns=['Geography','Gender'])

data.columns

## splitting the data...##
X=data.drop('Exited', axis=1)
y=data['Exited']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=10)


### Feature Scaling.....
from sklearn.preprocessing import StandardScaler ## multiplication will be easier(x1w1+b) and derivative will be faster in back propagation, which helps to converge first..##

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


## Importing Necessary Libraries...
import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


## Initializing ANN.....
clf=Sequential()

clf.add(Dense(units=30,kernel_initializer='he_uniform',activation='relu',input_dim=13)) ## relu works better with 'he_unifrom' weight initalizer..

##Adding 2nd hidden layer
clf.add(Dense(units=15,kernel_initializer='he_uniform',activation='relu'))
##Adding 3rd hidden layer
clf.add(Dense(units=7,kernel_initializer='he_uniform',activation='relu')) ## all hidden layers must have relu, leakyrelu...

## o/p  Layer
clf.add(Dense(units=1,kernel_initializer='glorot_normal',activation='sigmoid')) ## we can use kernel_initializer='glorot_uniform'...

## Compilng ANN...##
clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) ##  ADAM is the latest best optimizer..

## Now lets fit the ANN to training set...##

model= clf.fit(X_train, y_train, validation_split=0.2, batch_size=20, nb_epoch=100) # batch_size used to reduce the computational power, ram will be free.. can run on CPU ..

## Predicting on test data..
y_pred=clf.predict(X_test)
y_pred=(y_pred>0.5)

## Accuracy...
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)  ## 85%

## Classification Report...
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred)) 




########### Catboost Classifier..########################
from catboost import CatBoostClassifier
cb=CatBoostClassifier()

model=cb.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)  ## 85.9%~~ 86%

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test,y_pred))



