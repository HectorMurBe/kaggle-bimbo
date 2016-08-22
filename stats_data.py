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


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y+1)-np.log1p(y0+1), 2)))

def give_data(path_train,path_test):
    train_types = {'Semana':np.int8, 'Agencia_ID':np.int16,
             'Ruta_SAK':np.int16, 'Cliente_ID':np.int32, 'Producto_ID':np.int32,
             'Demanda_uni_equil':np.int16}
    test_types = {'Semana':np.int8,'Agencia_ID':np.int16, 'Ruta_SAK':np.int32, 'Cliente_ID':np.int32,
                      'Producto_ID':np.int16, 'id':np.int16}
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
        ant_ant.drop(["Semana","meanP","meanC","meanPA","meanPR","meanPC"],inplace=True,axis=1)#Drop duplicate column
        ant.drop(["Semana","meanP","meanC","meanPA","meanPR","meanPC"],inplace=True,axis=1)#Drop duplicate column
        ant_ant.columns=[u'Agencia_ID', u'Ruta_SAK', u'Cliente_ID', u'Producto_ID',u'l2']
        ant.columns=[u'Agencia_ID', u'Ruta_SAK', u'Cliente_ID', u'Producto_ID',u'l1']
        #First merge
        act=pd.merge(act,ant,how="left",on=["Agencia_ID","Ruta_SAK","Cliente_ID","Producto_ID"])
        #Second merge
        act=pd.merge(act,ant_ant,how="left",on=["Agencia_ID","Ruta_SAK","Cliente_ID","Producto_ID"])
        ordered_weeks.append(act.fillna(0))
    ordered_weeks.reverse()
    data=pd.concat(ordered_weeks[:-1])
    return data,ordered_weeks[-1]

def prepare_train_data(data,size=1000000):
    verbose('Isolating weeks')
    weeks=[]
    for i in range(3,9):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])
    temp=data[data.Semana==9].head(size)
    temp["Demanda_uni_equil"]=0
    weeks.append(temp)
    data=pd.concat(weeks)
    #test=weeks[-1]



    data['meanP']   = data.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanP")
    data['meanC']   = data.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanC")
    data['meanPA']  = data.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPA")
    data['meanPR']  = data.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPR")
    data['meanPC'] = data.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPC")
    verbose(data.info(memory_usage=True))


    data,test=preproc_weeks(data)
    train_labels=data.Demanda_uni_equil.values
    test_labels=test.Demanda_uni_equil.values
    data.drop(["Producto_ID","Ruta_SAK","Cliente_ID","Agencia_ID","Semana","Demanda_uni_equil"],inplace=True,axis=1)#Not usefull columns from here
    test.drop(["Producto_ID","Ruta_SAK","Cliente_ID","Agencia_ID","Semana","Demanda_uni_equil"],inplace=True,axis=1)#Not usefull columns from here
    return data.as_matrix(),train_labels,test.as_matrix(),test_labels

""" POSSIBLE SOLUTION FOR TEST
    test_=data.loc[:,['Producto_ID',"meanP"]]
    test=pd.merge(test,test_,how="left",on=["Producto_ID"])
    test_=data.loc[:,['Cliente_ID',"meanPC"]]
    test=pd.merge(test,test_,how="left",on=["Cliente_ID"])
    test_=data.loc[:,['Producto_ID','Agencia_ID',"meanPA"]]
    test=pd.merge(test,test_,how="left",on=['Producto_ID','Agencia_ID'])
    test_=data.loc[:,['Producto_ID','Ruta_SAK',"meanPR"]]
    test=pd.merge(test,test_,how="left",on=['Producto_ID','Ruta_SAK'])
    test_=data.loc[:,['Producto_ID','Cliente_ID',"meanPC"]]
    test=pd.merge(test,test_,how="left",on=['Producto_ID','Cliente_ID'])

    #remove excess columns... in final version won't ve neaded"""


def verbose(*args,**kargs):
    print(*args,**kargs)


def train_test(features,labels,features_test,labels_test):
    verbose ("Features size",features.shape)
    verbose ("Labels size",labels.shape)
    verbose ("Features size",features_test.shape)
    verbose ("Labels size",labels_test.shape)
    regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    verbose ("Training...")
    regressor.fit(features, labels)
    verbose ("Predict...")
    preds=regressor.predict(features_test)
    preds[preds<0]=0
    verbose(preds)
    verbose(len(preds))
    verbose("MSE: ")
    score = metrics.mean_squared_error(preds,labels_test)
    verbose("Original",score)
    score = metrics.mean_squared_error(np.round(preds), labels_test)
    verbose("round",score)
    verbose ("RMSLE:")
    score = rmsle(preds, labels_test)
    verbose ("Original",score)
    score = rmsle(np.round(preds), labels_test)
    verbose ("round",score)


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
    print("Reading data... ")
    df_train,df_test=give_data(opts.train,opts.test)
    print ("All data readed... ")
    features,labels,features_test,labels_test=prepare_train_data(df_train)
    train_test(features,labels,features_test,labels_test)

