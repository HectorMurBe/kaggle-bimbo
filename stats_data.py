#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import numpy as np
import pandas as pd
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics,preprocessing
def temp(data,size):
    weeks=[]
    for i in range(10,12):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)
    data["Demanda_uni_equil"]=0
    return data

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
    types = {'id':np.int8,'Semana':np.int8, 'Agencia_ID':np.int16,
             'Ruta_SAK':np.int16, 'Cliente_ID':np.int32, 'Producto_ID':np.int32 }
    return pd.read_csv("./data/test.csv", usecols=types.keys(), dtype=types)

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
def extract_span2(weeks,ix_pred,span):
    verbose("Processing week:",ix_pred)
    first=True
    idx_data={}
    f_=np.zeros([weeks[ix_pred-3].shape[0],span-1+5])
    verbose("Creating matrix for week",f_.shape)
    semana=ix_pred-3
    for val,row in enumerate(weeks[semana].itertuples()):
        idx_data[row[2:6]]=val
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
    return f_

def prepare_train_data(data,test,size=100000):
    verbose('Isolating weeks')
    weeks=[]
    for i in range(3,10):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])

    data=pd.concat(weeks)
    #print (data.tail(50))


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
    ids=data[data.tst==1]["id"].values
    data.drop(["tst","id"],inplace=True,axis=1)
    weeks=[]
    for i in range(3,12):
        weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)
    print (data.tail(50))
    del data
    print (ids)
    return weeks,ids
def predweeks(weeks,regressor,span=3):
    features=extract_span2(weeks[:-1],5,span)
    predictions10=predict(features,regressor)
    print (predictions10)
    weeks[-2]["Demanda_uni_equil"]=predictions10
    features=extract_span2(weeks[1:],5,span)
    predictions20=predict(features,regressor)
    print (predictions20)
    return np.concatenate((predictions10, predictions20))

def writesubmision(ids,predictions,path="./data/sumbision.csv"):
    submision=pd.DataFrame({"Demanda_uni_equil":predictions,"id":ids.astype("int8")})
    submision.to_csv(path,index=False)

def getraindata(weeks,span=3):
    verbose('Extracting span')
    feats=[]
    labels=[]
    for i in range(3+span-1,10):
        feats_,labels_=extract_span(weeks,i,span)
        feats.append(feats_)
        labels.append(labels_)

    return np.vstack(feats),np.hstack(labels)
def train(features,labels):
    regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    verbose ("Training...")
    regressor.fit(features, labels)
    return regressor
def predict(features,regressor):
    preds=regressor.predict(features)
    preds[preds<0]=0
    return preds

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
    df_train["tst"]=0
    verbose("Loading test data...")
    df_test=load_test(opts.train)
    df_test["tst"]=1
    df_test=temp(df_test)
    print (df_test.head(20))
    weeks,ids=prepare_train_data(df_train,df_test,size=opts.size)
    features,labels=getraindata(weeks[:-2],span=3)
    regressor=train(features,labels)
    predictions=predweeks(weeks[-4:],regressor)
    print ("Pedicciones juntas: ")
    print (predictions)
    writesubmision(ids,predictions,path="./data/sumbision.csv")
