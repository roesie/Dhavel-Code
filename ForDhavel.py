#!/usr/bin/env python3 -W ignore::DepreciationWarning
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:52:37 2017

@author: Mark
"""
# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
import sys

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

#%%
def get_accuracy_from_roc_curve(actuals, predictions):
    thresholds = np.linspace(0, 1, 1000)
    pre_list = []
    for i in range(len(thresholds)):
        thr = thresholds[i]
        pre = [*map(lambda x: 1 if x >= thr else 0, predictions)]
        pre_list.append(np.mean(actuals == pre))
        
    return max(pre_list)

#%%
# Importing the Dataset
data = pd.read_csv('churn.csv')

#Removing 'Unnamed: 0' and 'Phone' columns
data.drop('Unnamed: 0',axis = 1, inplace = True)
data.drop('Phone',axis = 1, inplace = True)
data.drop('Area Code', axis = 1, inplace = True) # Area code doesn't appear to be useful

# Raplce ? so label encoding works
data['VMail Plan'].replace('?','no',inplace=True)
data['Int\'l Plan'].replace('?','no',inplace=True)

le = LabelEncoder()
data['VMail Plan'] = le.fit_transform(data['VMail Plan'])
data['Int\'l Plan'] = le.fit_transform(data['Int\'l Plan'])

#%%
# Check for ?, replace with NaN (float value)
data.replace('?', 0, inplace=True)

# There are some large values which could be concerning
col_names = ['VMail Message', 'Account Length', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge','CustServ Calls']
data[col_names] = data[col_names].astype(float)
for col in col_names:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Create your train and test data sets
data['Churn?'] = data['Churn?'].apply(lambda x: x.split('.')[0])
y = le.fit_transform(data['Churn?'])
data.drop('Churn?', axis=1, inplace=True)

state = pd.get_dummies(data['State'])
data.drop('State', axis=1, inplace=True)
X = data.as_matrix() #pd.concat([data, state], axis=1).as_matrix()

#%%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

results = pd.DataFrame(columns=['AUC','units','dropout','batchsize','epoch'])    

# AUC .898 600 units, dropout 0.5, batch 150, epoch 150
units = [625, 650, 675] #def 600
dropout_rate = [0.55, 0.6, 0.65] #def 0.5
batch_size = [100, 125, 150] #def 150
epochs = [125, 175] #def 150
    
n = 1
for un in range(len(units)):
    for dr in range(len(dropout_rate)):
        for bs in range(len(batch_size)):
            for ep in range(len(epochs)):
                sys.stdout.flush()
                sys.stdout.write('\r{}/{}'.format(n,(len(units)*len(dropout_rate)*len(batch_size)*len(epochs))))
                
                prob_y = np.zeros_like(y,dtype='float')
                eval_y = np.zeros_like(y,dtype='float')
                
                for train, test in kfold.split(X, y):   
                    model = Sequential()    
                    model.add(Dense(units[un], activation='relu', kernel_initializer='random_normal', input_shape=(X[train].shape[1],)))
                    model.add(Dropout(dropout_rate[dr]))
                    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
                    model.fit(X[train], y[train], batch_size=batch_size[bs], epochs=epochs[ep], verbose=0)
                    
                    prob_y[test] = model.predict(X[test], verbose=0)
                    eval_y[test] = model.predict_classes(X[test], verbose=0) 
                    
                    del model                        
                #print('AUC: {:.3f}\tAcc: {:.1f}%'.format(roc_auc_score(y, prob_y), np.mean(y == eval_y)*100))
                n += 1
                results = results.append(pd.DataFrame([[roc_auc_score(y, prob_y), units[un], dropout_rate[dr], batch_size[bs], epochs[ep]]], columns=['AUC','units','dropout','batchsize','epoch']))
                           
print('{}'.format(results[results['AUC'] == max(results['AUC'])]))
