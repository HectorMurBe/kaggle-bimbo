#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import numpy as np
import pandas as pd
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics,preprocessing
from sklearn.svm import SVR
from  sklearn.ensemble import RandomForestRegressor
def temp(data,size):
    weeks=[]
    for i in range(10,12):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)
    data["Demanda_uni_equil"]=0
    print (data.tail())
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
    types = {'id':np.int32,'Semana':np.int8, 'Agencia_ID':np.int16,
             'Ruta_SAK':np.int16, 'Cliente_ID':np.int32, 'Producto_ID':np.int32 }
    return pd.read_csv("./data/test.csv", usecols=types.keys(), dtype=types)

def extract_span(weeks,ix_pred,span,ini=3):
    verbose("Processing week:",ix_pred)
    first=True
    idx_data={}
    f_=np.zeros([weeks[ix_pred-ini].shape[0],span-1+5])
    l_=np.zeros(weeks[ix_pred-ini].shape[0])
    verbose("Creating matrix for week",f_.shape)
    semana=ix_pred-3

    verbose(weeks[semana].info(memory_usage=True))
    for val,row in enumerate(weeks[semana].itertuples()):
        idx_data[row[2:6]]=val
        l_[val]=row[6]
        f_[val,span-1]=row[7]
        f_[val,span]=row[8]
        f_[val,span+1]=row[9]
        f_[val,span+2]=row[10]
        f_[val,span+3]=row[11]
    for i in range(span)[1:span]:
        semana=ix_pred-i-ini
        verbose("Adding info week",semana+ini)
        for row in weeks[semana].itertuples():
            try:
                val=idx_data[row[2:6]]
                f_[val,span-i-1]=row[6]
            except:
                pass
    return f_,l_

def extract_span2(weeks,ix_pred,span=3,ini=3):
    verbose("Processing week:",ix_pred)
    first=True
    idx_data={}
    f_=np.zeros([weeks[ix_pred-ini].shape[0],span-1+5])
    verbose("Creating matrix for week",f_.shape)
    semana=ix_pred-ini
    ids=[]
    for val,row in enumerate(weeks[semana].itertuples()):
        idx_data[row[2:6]]=val
        f_[val,span-1]=row[7]
        f_[val,span]=row[8]
        f_[val,span+1]=row[9]
        f_[val,span+2]=row[10]
        f_[val,span+3]=row[11]
        ids.append(row[12])
    for i in range(span)[1:span]:
        semana=ix_pred-i-ini
        verbose("Adding info week",semana+ini)
        for row in weeks[semana].itertuples():
            try:
                val=idx_data[row[2:6]]
                f_[val,span-i-1]=row[6]
            except:
                pass
    return f_,ids

def prepare_train_data(data,test,size=100000,ini=3):
    verbose('Isolating weeks')
    weeks=[]
    weeks_id=data.Semana.unique()
    for i in range(ini,ini+len(weeks_id)):
        weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)

    """
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
    """
    summary={}
    summary['meanP']   = data.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanP")
    summary['meanC']   = data.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanC")
    summary['meanPA']  = data.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPA")
    summary['meanPR']  = data.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPR")
    summary['meanPC'] =  data.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated meanPC")
    verbose('Isolating weeks (again)')
    weeks=[]
    for i in range(ini,10):
        if size and i!=9:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])
    data=pd.concat(weeks)

    data['meanP']=summary['meanP']
    data['meanC']=summary['meanC']
    data['meanPA']=summary['meanPA']
    data['meanPR']=summary['meanPR']
    data['meanPC']=summary['meanPC']

    test['meanP']=summary['meanP']
    test['meanC']=summary['meanC']
    test['meanPA']=summary['meanPA']
    test['meanPR']=summary['meanPR']
    test['meanPC']=summary['meanPC']

    for k,v in summary.iteritems():
        del v

    '''test_ = data.loc[:, ['Producto_ID', 'meanP']].drop_duplicates(subset=['Producto_ID'])
    test = test.merge(test_,how='left',on=['Producto_ID'],copy=False)
    test_ = data.loc[:, ['Cliente_ID', 'meanC']].drop_duplicates(subset=['Cliente_ID'])
    test = test.merge(test_,how='left',on=['Cliente_ID'],copy=False)
    test_ = data.loc[:, ['Producto_ID','Agencia_ID','meanPA']].drop_duplicates(subset=['Producto_ID','Agencia_ID'])
    test = test.merge(test_,how='left',on=['Producto_ID','Agencia_ID'],copy=False)
    test_ = data.loc[:, ['Producto_ID','Ruta_SAK','meanPR']].drop_duplicates(subset=['Producto_ID','Ruta_SAK'])
    test = test.merge(test_,how='left',on=['Producto_ID','Ruta_SAK'],copy=False)
    test_ = data.loc[:, ['Producto_ID','Cliente_ID','meanPC']].drop_duplicates(subset=['Producto_ID','Cliente_ID'])
    test = test.merge(test_,how='left',on=['Producto_ID','Cliente_ID'],copy=False)
    del test_'''

    verbose("Calculated meanPC")
    test.fillna(0, inplace=True)
    print ("Test shape: ")
    print (test.shape)
    print ("Valores con 0 en Cliente:")
    print (test[test.meanC == 0.0].count(axis=1).shape)
    print (test["meanC"].value_counts(dropna=False)[0])
    print ("Valores con 0 en Producto:")
    print (test[test.meanP == 0.0].count(axis=1).shape)
    print (test["meanP"].value_counts(dropna=False)[0])
    print ("Valores con 0 en Producto Agencia:")
    print (test[test.meanPA == 0.0].count(axis=1).shape)
    print (test["meanPA"].value_counts(dropna=False)[0])
    print ("Valores con 0 en Producto Cliente:")
    print (test[test.meanPC == 0.0].count(axis=1).shape)
    print (test["meanPC"].value_counts(dropna=False)[0])
    print ("Valores Clientes Unicos:")
    print (len(np.setdiff1d(test["Cliente_ID"].values,data["Cliente_ID"])))
    print ("Valores Productos Unicos:")
    print (len(np.setdiff1d(test["Producto_ID"].values,data["Producto_ID"])))
    print ("Valores Ruta_SAK Unicos:")
    print (len(np.setdiff1d(test["Ruta_SAK"].values,data["Ruta_SAK"])))
    print ("Valores Agencia_ID Unicos:")
    print (len(np.setdiff1d(test["Agencia_ID"].values,data["Agencia_ID"])))


    data=pd.concat([data,test])

    ids=[]
    verbose(data.info(memory_usage=True))
    data.drop(["tst"],inplace=True,axis=1)
    verbose ("Data before order")
    data=data[["Semana",'Agencia_ID','Cliente_ID','Producto_ID','Ruta_SAK','Demanda_uni_equil','meanP','meanC','meanPA',"meanPR",'meanPC','id']]
    verbose(data.info(memory_usage=True))
    weeks=[]
    for i in range(3,12):
        weeks.append(data[data.Semana==i])
    return weeks,ids

def predweeks(weeks,regressor,span=3):
    features=extract_span2(weeks[:-1],5,span)
    predictions10=predict(features,regressor)
    weeks[-2]["Demanda_uni_equil"]=predictions10
    features=extract_span2(weeks[1:],5,span)
    predictions20=predict(features,regressor)
    print (predictions20)
    return np.concatenate((predictions10, predictions20))

def writesubmision(ids,predictions,path="./data/submission.csv"):
    submision=pd.DataFrame({"id":ids.astype("int32"),"Demanda_uni_equil":predictions})
    submision=submision['id','Demanda_uni_equil']
    submision.to_csv(path,index=False)

def getraindata(weeks,span=3,ini=3):
    verbose('Extracting span')
    feats=[]
    labels=[]
    for i in range(ini+span-1,len(weeks)+ini):
        feats_,labels_=extract_span(weeks,i,span)
        feats.append(feats_)
        labels.append(labels_)
    return np.vstack(feats),np.hstack(labels)

def train(features,labels):
    verbose ("Features size",features.shape)
    verbose ("Labels size",labels.shape)
    #regressor = skflow.TensorFlowLinearRegressor(steps=2000,learning_rate=0.1)#TODO convert uint32 to TensorFlow DType
    regressor = RandomForestRegressor(n_estimators=20)
    verbose ("Training...")
    regressor.fit(features, labels)
    return regressor

def predict(features,regressor):
    verbose ("Size test",features.shape)
    preds=regressor.predict(features)
    preds[preds<0]=1
    return preds

def eval_test(preds,labels):
    verbose(len(preds))
    verbose("MSE: ")
    score = metrics.mean_squared_error(preds, labels)
    verbose("Original",score)
    score = metrics.mean_squared_error(np.round(preds), labels)
    verbose("round",score)
    verbose ("RMSLE:")
    score = rmsle(preds, labels)
    verbose ("Original",score)
    score = rmsle(np.round(preds), labels)
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
    p.add_argument("--size",default=None,type=int,
            action="store", dest="size",
            help="Size of records per week [all]")
    p.add_argument("--span",default=3,type=int,
            action="store", dest="span",
            help="Span of weeks")
    p.add_argument("-d", "--dev",
            action="store_true", dest="dev",
            help="Development mode [Off]")
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
    df_test=temp(df_test,False)
    weeks,ids=prepare_train_data(df_train,df_test,size=opts.size)
    if opts.dev:
        features,labels=getraindata(weeks[:-3],span=opts.span)
        regressor=train(features,labels)
        feats,labels=extract_span(weeks,9,span=opts.span)
        predictions=predict(feats,regressor)
        eval_test(predictions,labels)
    else:
        features,labels=getraindata(weeks[:-2],span=opts.span)
        regressor=train(features,labels)
        feats,ids1=extract_span2(weeks,10,span=opts.span)
        predictions10=predict(feats,regressor)
        verbose (predictions10)
        weeks[-2]["Demanda_uni_equil"]=predictions10
        feats,ids2=extract_span2(weeks,11,span=opts.span)
        predictions11=predict(feats,regressor)
        verbose (predictions11)
        predictions=np.concatenate((predictions10, predictions11))
        ids=np.array(ids1+ids2)
        verbose ('predictions', predictions.shape)
        verbose ('ids', ids.shape)
        writesubmision(ids,predictions)
