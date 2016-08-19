#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import argparse
import pandas as pd
import numpy as np
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics,preprocessing
#import models_tutogithub as tflinear


def give_data(path_train,path_test):
    train_types = {'Semana':np.uint32,'Agencia_ID':np.uint32, 'Ruta_SAK':np.uint32, 'Cliente_ID':np.uint32,'Producto_ID':np.uint32, 'Demanda_uni_equil':np.uint32}

    test_types = {'Semana':np.uint32,'Agencia_ID':np.uint32, 'Ruta_SAK':np.uint32, 'Cliente_ID':np.uint32,
                      'Producto_ID':np.uint16, 'id':np.uint16}
    df_train = pd.read_csv(path_train, usecols=train_types.keys(), dtype=train_types)
    df_test = pd.read_csv(path_test,usecols=test_types.keys(), dtype=test_types)
    return df_train,df_test
def temp_preproc_weeks(dataframe,size):
    weeks=[]
    for i in np.unique(dataframe.Semana):
        weeks.append(dataframe[dataframe.Semana==i].sample(size).groupby(["Agencia_ID","Ruta_SAK","Cliente_ID","Producto_ID"])["Demanda_uni_equil"])
        print (weeks[i])
    return weeks
def preproc_weeks(dataframe):
    weeks=[]
    for i in np.unique(dataframe.Semana):
        weeks.append(df_train[dataframe.Semana==i])
        #TODO:ORDER DATA by cliente,ruta,producto; Weeks have diferent sizes
    return weeks
"""def data_preproces(weeks,logsize):
    features=[]
    labels=[]
    for i in  range(len(weeks)-logsize-1):
        weeks[i]=pd.concat(weeks[i:(i+logsize)],axis=1)
    return weeks
    PROBLEMA AL CONCATENAR
    """
def data_preproces(weeks):
    weeks['MeanP'] = weeks.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanP')
    weeks['MeanC'] = weeks.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanC')
    features=[]
    labels=[]
    for i in range(4,9):
        mini=pd.concat([df_train[df_train.Semana==(i-1)].head(1000),df_train[df_train.Semana==(i)].head(1000),df_train[df_train.Semana==(i+1)].head(1000)])
        group=mini.groupby(["Cliente_ID","Agencia_ID","Producto_ID","Ruta_SAK","MeanP","MeanC"])#TODO: change mini for weeks
        for group,index in zip(group.groups.keys(),group.groups.values()):
            stds=[0,0]
            val_weeks=np.unique(weeks.loc[index].Semana)
            val_std=mini.loc[index].Demanda_uni_equil.as_matrix()
            if (i+1) in val_weeks:
                print ("LABEL: ")
                print (val_std[-1])
                labels.append(val_std[-1])
                val_std=val_std[:-1]
                val_weeks=val_weeks[:-1]
                print ("val_weeks:")
                print (val_weeks)
                print ("std:")
                print (val_std)
                for week,std in zip(val_weeks,val_std):
                    stds[week%(i-1)]=std
                print ("group keys:")
                print (group)
                new_feature=np.concatenate((group,stds)).astype("float32")
                print ("New features:")
                print (new_feature)




    return np.array(features),np.array(labels).astype("float32")



def old_data_preproces(weeks,logsize):
    #dataframe to matrix
    for i in range(len(weeks)):
        weeks[i]=weeks[i].as_matrix()
    features=[]
    labels=[]#TODO implement with numpy
    for i in range(len(weeks)-logsize):
        weeks[i]=np.concatenate(weeks[i:(i+logsize)], axis=1)
        #weeks[i]=np.concatenate([weeks[i],weeks[i+logsize][][:-1]],axis=1)
        for j in range(len(weeks[i])):
            #print (weeks[i][j])
            #print (weeks[i+logsize][j][:-1])
            feature=np.concatenate([weeks[i][j],weeks[i+logsize][j][:-1]])
            features.append(feature)
            labels.append(weeks[i+logsize][j][-1])
        print (weeks[i])
    del weeks
    #print (features)
    #print (labels)
    return np.array(features),np.array(labels)

def model(features,labels,test_size):

    regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    regressor.fit(features[:-test_size], labels[:-test_size])
    score = metrics.mean_squared_error(regressor.predict(features[-test_size:]), labels[-test_size:])
    print ("MSE: ")
    print (score)
    print (features[-1])
    print (labels[-1])
    print (regressor.predict(np.array([features[-1]])))
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
    features,labels=data_preproces(df_train)
    print ("Starting train")
    model(features,labels,2000)
    print ("END :D!")
    #weeks=preproc_weeks(df_train)
    """
    features,labels=data_preproces(weeks,2)
    features = preprocessing.StandardScaler().fit_transform(features)
    func_label = preprocessing.StandardScaler().fit(labels)
    labels_=func_label.transform(labels)
    print ("Starting train")
    model(features,labels,labels_,func_label,20)
    print ("END :D!")
    print("Tf model:")

    tflinear.LinerReg(features,labels,20,17)




    pd.concat([df_train[df_train.Semana==3].head(1000),df_train[df_train.Semana==4].head(1000),df_train[df_train.Semana==5].head(1000)])
     group=mini.groupby(["Cliente_ID","Agencia_ID","Producto_ID","Ruta_SAK","MeanP","MeanC"])"""
