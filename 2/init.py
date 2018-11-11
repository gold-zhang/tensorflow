# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:51:13 2018

@author: yao
"""

from sklearn import preprocessing
from sklearn import datasets
from numpy import *

def normalization(data,target):
    min_max_scaler = preprocessing.MinMaxScaler()  
    data = min_max_scaler.fit_transform(data)
    label = zeros([150,3])
    for i in range(150):
        label[i][target[i]] = 1
    return data,label

def loadData():
    iris = datasets.load_iris()
    #n_samples,n_features=iris.data.shape
    #print("Number of sample:",n_samples)  
    #print("Number of feature",n_features)
    data = iris.data
    label = iris.target
    data,label = normalization(data,label)
    return data,label


