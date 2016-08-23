#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import numpy as np
import pandas as pd
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics,preprocessing
def temp(test,size):
    for i in range(10,11):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)

def verbose(*args,**kargs):
    print(*args,**kargs)

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y+1)-np.log1p(y0+1), 2)))

def load_train(filepath):
    types = {'Semana':np.int8, 'Agencia_ID':np.int16,
             'Ruta_SAK':np.int16, 'Cliente_ID':np.int32, 'Producto_ID':np.int32,
             'Demanda_uni_equil':np.int16}
    return pd.read_csv(filepath, usecols=types.keys(), dtype=types)
def load_test(filepath):
    types = {'Semana':np.int8, 'Agencia_ID':np.int16,
             'Ruta_SAK':np.int16, 'Cliente_ID':np.int32, 'Producto_ID':np.int32,
             'Demanda_uni_equil':np.int8}
    return pd.read_csv(filepath, usecols=types.keys(), dtype=types)

def extract_span(weeks,ix_pred,span):
    verbose("Processing week:",ix_pred)
    first=True
    idx_data={}
    f_=np.zeros([weeks[ix_pred-3].shape[0],span-1+5])
    l_=np.zeros(weeks[ix_pred-3].shape[0])
    verbose("Creating matrix for week",f_.shape)
    semana=ix_pred-3
    for val,row in enumerate(weeks[semana].itertuples()):
        idx_data[row[2:6]]=val
        l_[val]=row[6]
        f_[val,span-1]=row[7]
        f_[val,span]=row[8]
        f_[val,span+1]=row[9]
        f_[val,span+2]=row[10]
        f_[val,span+3]=row[11]
    for i in range(span)[1:span]:
        semana=ix_pred-i-3
        verbose("Adding info week",semana+3)
        for row in weeks[semana].itertuples():
            try:
                val=idx_data[row[2:6]]
                f_[val,span-i-1]=row[6]
            except:
                pass
    return f_,l_

def prepare_train_data(data,span=3,size=100000):
    verbose('Isolating weeks')
    weeks=[]
    for i in range(3,9):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])

    test=data[data.Semana==9].head(size)
    print (test.head())
    data=pd.concat(weeks)
    summary={}
    data['meanP']   = data.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    test['meanP']   = data.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanP")
    data['meanC']   = data.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    test['meanC']   = data.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanC")
    data['meanPA']  = data.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    test['meanPA']  = data.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPA")
    data['meanPR']  = data.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    test['meanPR']  = data.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPR")
    data['meanPC'] = data.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform('mean')
    test['meanPC'] = data.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform('mean')
    verbose("Calculated meanPC")
    test.fillna(0, inplace=True)
    data=pd.concat([data,test])
    print (data.tail())
    

    verbose(data.info(memory_usage=True))

    verbose('Isolating weeks (again)')
    weeks=[]
    for i in range(3,10):
        weeks.append(data[data.Semana==i])
    test_size=weeks[-1].shape[0]
    data=pd.concat(weeks)

    del data

    verbose('Extracting span')
    feats=[]
    labels=[]
    for i in range(3+span-1,10):
        feats_,labels_=extract_span(weeks,i,span)
        feats.append(feats_)
        labels.append(labels_)

    return np.vstack(feats),np.hstack(labels),test_size

def train_test(features,labels,test_size):
    verbose ("Features size",features.shape)
    verbose ("Labels size",labels.shape)
    verbose ("Size test",test_size)
    regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    verbose ("Training...")
    regressor.fit(features[:-test_size], labels[:-test_size])
    verbose ("Predict...")
    preds=regressor.predict(features[-test_size:])
    preds[preds<0]=0
    verbose(preds)
    verbose(len(preds))
    verbose("MSE: ")
    score = metrics.mean_squared_error(preds, labels[-test_size:])
    verbose("Original",score)
    score = metrics.mean_squared_error(np.round(preds), labels[-test_size:])
    verbose("round",score)
    verbose ("RMSLE:")
    score = rmsle(preds, labels[-test_size:])
    verbose ("Original",score)
    score = rmsle(np.round(preds), labels[-test_size:])
    verbose ("round",score)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser("Statistics data")
    p.add_argument("--train",default="data/train.csv",
            action="store", dest="train",
            help="Train file [train.csv]")
    p.add_argument("--test",default="data/test.csv",
            action="store", dest="test",
            help="Train file [test.csv]")
    p.add_argument("--test_size",default=1000,type=int,
            action="store", dest="test_size",
            help="Size of records per week [all]")
    p.add_argument("--size",default=4000000,type=int,
            action="store", dest="size",
            help="Size of records per week [all]")
    p.add_argument("--span",default=3,type=int,
            action="store", dest="span",
            help="Span of weeks")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()

    # Prepara función de verbose  -----------------------------------------
    if not opts.verbose:
       verbose = lambda *a: None

    verbose("Loading training data...")
    df_train=load_train(opts.train)
    """verbose("Loading test data...")
    df_test=load_train(opts.train)
    df_test=temp(df_test,10)
    print (df_test[0].head(10))
    print (df_test[1].head(10))"""
    verbose("All data readed...")
    verbose(df_train.info(memory_usage=True))
    feats,labels,test_size=prepare_train_data(df_train,size=opts.size)

    train_test(feats,labels,test_size)
