#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import argparse
import pandas as pd
import numpy as np
import tensorflow.contrib.learn as skflow
import tensorflow as tf

def give_data(path_train,path_test):
    train_types = {'Semana':np.uint8,'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32,'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}

    test_types = {'Semana':np.uint8,'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32,
                  'Producto_ID':np.uint16, 'id':np.uint16}
    df_train = pd.read_csv(path_train, usecols=train_types.keys(), dtype=train_types)
    df_test = pd.read_csv(path_test,usecols=test_types.keys(), dtype=test_types)
    return df_train,df_test

def temp_preproc_weeks(dataframe,size):
    weeks=[]
    for i in np.unique(dataframe.Semana):
        weeks.append(df_train[dataframe.Semana==i].sample(size))
    return weeks

def preproc_weeks(dataframe):
    weeks=[]

    for i in np.unique(dataframe.Semana):
        weeks.append(df_train[dataframe.Semana==i])
        #TODO:ORDER DATA by cliente,ruta,producto
    return weeks
"""def data_preproces(weeks,logsize):
    features=[]
    labels=[]
    for i in  range(len(weeks)-logsize-1):
        weeks[i]=pd.concat(weeks[i:(i+logsize)],axis=1)
    return weeks
    PROBLEMA AL CONCATENAR
    """

def data_preproces(weeks,logsize):
    #dataframe to matrix
    for i in range(len(weeks)):
        weeks[i]=weeks[i].as_matrix()
    features=[]
    labels=[]#TODO implement with numpy
    for i in range(len(weeks)-logsize):
        weeks[i]=np.concatenate(weeks[i:(i+logsize)], axis=1)
        #weeks[i]=np.concatenate([weeks[i],weeks[i+logsize][][:-1]],axis=1)
        for j in range(len(weeks[i])):
            print (weeks[i][j])
            print (weeks[i+logsize][j][:-1])
            feature=np.concatenate([weeks[i][j],weeks[i+logsize][j][:-1]])
            features.append(feature)
            labels.append(weeks[i+logsize][j][-1])
        print (weeks[i])
    del weeks
    print (features)
    print (labels)
    return np.array(features),np.array(labels)

def model(features,labels,test_size):
    regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    regressor.fit(features[:-test_size], labels[:-test_size])
    score = metrics.mean_squared_error(regressor.predict(features[-test_size:]), labels[-test_size:])
    print ("MSE: ")
    print (score)
    return regressor


if __name__ == '__main__':
    p = argparse.ArgumentParser("Statistics data")
    p.add_argument("--train",default="data/train.csv",
            action="store", dest="train",
            help="Train file [train.csv]")
    p.add_argument("--test",default="data/test.csv",
            action="store", dest="test",
            help="Train file [test.csv]")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()
    print("Reading data...")
    df_train,df_test=give_data(opts.train,opts.test)
    print ("All data readed...")
    weeks=temp_preproc_weeks(df_train,2)
    features,labels=data_preproces(weeks,2)
    features = tf.cast(features, tf.int32)
    model(features,labels,2)
