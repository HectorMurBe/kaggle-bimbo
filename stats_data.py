#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import argparse
import pandas as pd
import numpy as np
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics,preprocessing
from  sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
#import models_tutogithub as tflinear


def give_data(path_train,path_test):
    train_types = {'Semana':np.uint32,'Agencia_ID':np.uint32, 'Ruta_SAK':np.uint32, 'Cliente_ID':np.uint32,'Producto_ID':np.uint32, 'Demanda_uni_equil':np.uint32}

    test_types = {'Semana':np.uint32,'Agencia_ID':np.uint32, 'Ruta_SAK':np.uint32, 'Cliente_ID':np.uint32,
                      'Producto_ID':np.uint16, 'id':np.uint16}
    df_train = pd.read_csv(path_train, usecols=train_types.keys(), dtype=train_types)
    df_test = pd.read_csv(path_test,usecols=test_types.keys(), dtype=test_types)
    return df_train,df_test

def preproc_weeks(weeks):
    ordered_weeks=[]
    for i in reversed(range(5,10)):
        #Take the weeks you want to log
        ant_ant=weeks[weeks.Semana==i-2]
        ant=weeks[weeks.Semana==i-1]
        act=weeks[weeks.Semana==i]
        #Rename columns Demanda_uni_equil for log
        ant_ant.drop(["Semana"],inplace=True,axis=1)#Drop duplicate column
        ant.drop(["Semana"],inplace=True,axis=1)#Drop duplicate column
        ant_ant.columns=[u'Agencia_ID', u'Ruta_SAK', u'Cliente_ID', u'Producto_ID',u'l2']
        ant.columns=[u'Agencia_ID', u'Ruta_SAK', u'Cliente_ID', u'Producto_ID',u'l1']
        ant.rename(columns = {'Demanda_uni_equil':'l1'})
        #First merge
        act=pd.merge(act,ant,how="left",on=["Agencia_ID","Ruta_SAK","Cliente_ID","Producto_ID"])
        #Second merge
        act=pd.merge(act,ant_ant,how="left",on=["Agencia_ID","Ruta_SAK","Cliente_ID","Producto_ID"])
        ordered_weeks.append(act.fillna(0))
    print (ordered_weeks[0].head())
    return ordered_weeks






def data_preproces(weeks):
    size=10000


    weeks=pd.concat([
        weeks[weeks.Semana==(7)].head(size),
        weeks[weeks.Semana==(8)].head(size),
        weeks[weeks.Semana==(9)].head(size)])
    weeks['MeanP'] = weeks.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanP')
    weeks['MeanC'] = weeks.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanC')
    weeks['MeanA'] = weeks.groupby('Agencia_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanA')
    weeks['MeanR'] = weeks.groupby('Ruta_SAK')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    print ('Got MeanR')
    group=weeks.groupby(["Cliente_ID","Agencia_ID","Producto_ID","Ruta_SAK","MeanP","MeanC","MeanA","MeanR"])#TODO: change mini for weeks
    features=[]
    print ('Got groups')
    labels=[]
    for i in range(8,9):
        print ('i',i)
        for group,index in group:
            #print ('index',index.index)
            stds=[0,0]
            val_weeks=index.Semana.as_matrix()
            val_std=index.Demanda_uni_equil.as_matrix()
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
                print (group[-4:])
                print (stds)
                new_feature=np.concatenate((group[-4:],stds)).astype("float32")
                print ("New features:")
                print (new_feature)
                features.append(new_feature)

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
    #regressor = SVR()
    #regressor = LinearRegression()
    print(features[:-test_size].shape)
    regressor.fit(features[:-test_size], labels[:-test_size])
    preds=np.round(regressor.predict(features[-test_size:]))
    print(preds[:2],labels[-test_size:][:2])
    score = metrics.mean_squared_error(preds, labels[-test_size:])
    print ("MSE: ")
    print (score)
    print (features[-1])
    print (labels[-1])
    print (regressor.predict(np.array([features[-1]])))
    return regressor


def model2(features,labels,test_size):
    features = preprocessing.StandardScaler().fit_transform(features)
    regressor = linear_model.LinearRegression()
    regressor.fit(features[:-test_size], labels[:-test_size])
    preds=regressor.predict(features[-test_size:])
    print(preds[:2],labels[-test_size:][:2])
    score = metrics.mean_squared_error(preds, labels[-test_size:])
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
    preproc_weeks(df_train)
    """
    features,labels=data_preproces(df_train)
    print ("Starting train")
    model(features,labels,200)
    print ("END :D!")
    print(features.shape)
    print(labels.shape)"""


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
