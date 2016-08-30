#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import numpy as np
import pandas as pd
#import tensorflow.contrib.learn as skflow
#import tensorflow as tf
from sklearn import metrics,preprocessing
import xgboost as xgb
import gc

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



def preproc_weeks(weeks):
    ordered_weeks=[]
    for i in reversed(range(5,10)):
        #Take the weeks you want to log
        ant_ant=weeks[weeks.Semana==i-2]
        ant=weeks[weeks.Semana==i-1]
        act=weeks[weeks.Semana==i]
        #ant=pd.concat([ant,ant_ant])
        """summary={}
        #summary['lg1_pc']   = ant.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg1_PC")
        summary['lg1_p']   = ant.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        verbose("Calculated lg1_P")
        summary['lg1_c']   = ant.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        verbose("Calculated lg1_C")
        #summary['lg1_pa']   = ant.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg1_PA")
        #summary['lg1_pr']   = ant.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg1_PR")
        #summary['lg2_pc']   = ant.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg2_PC")
        #summary['lg2_p']   = ant.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg2_P")
        #summary['lg2_c']   = ant.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg2_C")
        #summary['lg2_pa']   = ant.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg2_PA")
        #summary['lg2_pr']   = ant.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        #verbose("Calculated lg2_PR")
        #Second merge
        act['lg1_c']=summary['lg1_c']
        act['lg1_p']=summary['lg1_p']
        #act['lg1_pc']=summary['lg1_pc']
        #act['lg1_pa']=summary['lg1_pa']
        #act['lg1_pr']=summary['lg1_pr']
        #act['lg2_c']=summary['lg2_c']
        #act['lg2_p']=summary['lg2_p']
        #act['lg2_pc']=summary['lg2_pc']
        #act['lg2_pa']=summary['lg2_pa']
        #act['lg2_pr']=summary['lg2_pr']"""
        #First LOG
        aux=ant.loc[:, ['Producto_ID', 'Demanda_uni_equil']]
        aux['lg1_p']   = aux.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Producto_ID'],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Producto_ID'],copy=False)

        aux=ant.loc[:, ['Cliente_ID',"Producto_ID", 'Demanda_uni_equil']]
        aux['lg1_pc']   = aux.groupby(['Cliente_ID',"Producto_ID"])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Cliente_ID',"Producto_ID"],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Cliente_ID','Cliente_ID',"Producto_ID"],copy=False)

        aux=ant.loc[:, ['Cliente_ID', 'Demanda_uni_equil']]
        aux['lg1_c']   = aux.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Cliente_ID'],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Cliente_ID'],copy=False)

        #Second Log

        aux=ant_ant.loc[:, ['Producto_ID', 'Demanda_uni_equil']]
        aux['lg2_p']   = aux.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Producto_ID'],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Producto_ID'],copy=False)

        aux=ant_ant.loc[:, ['Cliente_ID',"Producto_ID", 'Demanda_uni_equil']]
        aux['lg2_pc']   = aux.groupby(['Cliente_ID',"Producto_ID"])['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Cliente_ID',"Producto_ID"],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Cliente_ID','Cliente_ID',"Producto_ID"],copy=False)

        aux=ant_ant.loc[:, ['Cliente_ID', 'Demanda_uni_equil']]
        aux['lg2_c']   = aux.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
        aux = aux.reset_index()
        aux=aux.drop_duplicates(subset=['Cliente_ID'],keep="first")
        aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
        act=pd.merge(act,aux,how='left',on=['Cliente_ID'],copy=False)
        print (act.head())
        ordered_weeks.append(act.fillna(0))
    ordered_weeks.reverse()
    data=pd.concat(ordered_weeks)
    return data
def preproc_weeks2(weeks,target):
    weeks=pd.concat(weeks)
    i=target
        #Take the weeks you want to log
    ant_ant=weeks[weeks.Semana==i-2]
    ant=weeks[weeks.Semana==i-1]
    act=weeks[weeks.Semana==i]
        #ant=pd.concat([ant,ant_ant])
    """summary={}
    #summary['lg1_pc']   = ant.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg1_PC")
    summary['lg1_p']   = ant.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated lg1_P")
    summary['lg1_c']   = ant.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    verbose("Calculated lg1_C")
    #summary['lg1_pa']   = ant.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg1_PA")
    #summary['lg1_pr']   = ant.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg1_PR")
    #summary['lg2_pc']   = ant_ant.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg2_PC")
    #summary['lg2_p']   = ant_ant.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg2_P")
    #summary['lg2_c']   = ant_ant.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg2_C")
    #summary['lg2_pa']   = ant_ant.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg2_PA")
    #summary['lg2_pr']   = ant_ant.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    #verbose("Calculated lg2_PR")
        #Second merge
    act['lg1_c']=summary['lg1_c']
    act['lg1_p']=summary['lg1_p']
    #act['lg1_pc']=summary['lg1_pc']
    #act['lg1_pa']=summary['lg1_pa']
    #act['lg1_pr']=summary['lg1_pr']
    #act['lg2_c']=summary['lg2_c']
    #act['lg2_p']=summary['lg2_p']
    #act['lg2_pc']=summary['lg2_pc']
    #act['lg2_pa']=summary['lg2_pa']
    #act['lg2_pr']=summary['lg2_pr']"""
    #First LOG
    aux=ant.loc[:, ['Producto_ID', 'Demanda_uni_equil']]
    aux['lg1_p']   = aux.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID'],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Producto_ID'],copy=False)

    aux=ant.loc[:, ['Cliente_ID',"Producto_ID", 'Demanda_uni_equil']]
    aux['lg1_pc']   = aux.groupby(['Cliente_ID',"Producto_ID"])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Cliente_ID',"Producto_ID"],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Cliente_ID','Cliente_ID',"Producto_ID"],copy=False)

    aux=ant.loc[:, ['Cliente_ID', 'Demanda_uni_equil']]
    aux['lg1_c']   = aux.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Cliente_ID'],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Cliente_ID'],copy=False)

    #Second Log

    aux=ant_ant.loc[:, ['Producto_ID', 'Demanda_uni_equil']]
    aux['lg2_p']   = aux.groupby('Producto_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID'],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Producto_ID'],copy=False)

    aux=ant_ant.loc[:, ['Cliente_ID',"Producto_ID", 'Demanda_uni_equil']]
    aux['lg2_pc']   = aux.groupby(['Cliente_ID',"Producto_ID"])['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Cliente_ID',"Producto_ID"],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Cliente_ID','Cliente_ID',"Producto_ID"],copy=False)

    aux=ant_ant.loc[:, ['Cliente_ID', 'Demanda_uni_equil']]
    aux['lg2_c']   = aux.groupby('Cliente_ID')['Demanda_uni_equil'].transform(np.mean).astype('float32')
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Cliente_ID'],keep="first")
    aux.drop(['index',"Demanda_uni_equil"],inplace=True,axis=1)
    act=pd.merge(act,aux,how='left',on=['Cliente_ID'],copy=False)
    print (act.head())
    act.fillna(0, inplace=True)
    act.drop(["Semana","Demanda_uni_equil"],inplace=True,axis=1)
    print (act.head())
    return act.as_matrix()

def prepare_train_data(data,test,size=100000):
    verbose('Isolating weeks')
    weeks=[]
    for i in range(3,10):
        if size:
            weeks.append(data[data.Semana==i].head(size))
        else:
            weeks.append(data[data.Semana==i])

    data=pd.concat(weeks)
    data['Demanda_uni_equil'] = np.log(data['Demanda_uni_equil'] + 1)
    test['Demanda_uni_equil'] = np.log(test['Demanda_uni_equil'] + 1)
    #print (data.tail(50))
    #mean by product
    mean_P =  pd.DataFrame({'meanP':df_train.groupby('Producto_ID')['Demanda_uni_equil'].mean()}).reset_index()

    mean_C =  pd.DataFrame({'meanC':df_train.groupby('Cliente_ID')['Demanda_uni_equil'].mean()}).reset_index()

    mean_PA = pd.DataFrame({'meanPA':df_train.groupby(['Producto_ID','Agencia_ID'])['Demanda_uni_equil'].mean()}).reset_index()

    mean_PR = pd.DataFrame({'meanPR':df_train.groupby(['Producto_ID','Ruta_SAK'])['Demanda_uni_equil'].mean()}).reset_index()

    mean_PC = pd.DataFrame({'meanPC':df_train.groupby(['Producto_ID','Cliente_ID'])['Demanda_uni_equil'].mean()}).reset_index()
    mean_P.drop_duplicates(subset=['Producto_ID'],keep="first")
    data=pd.merge(data,mean_P,how='left',on=['Producto_ID'],copy=False)
    mean_P.drop_duplicates(subset=['Cliente_ID'],keep="first")
    data=pd.merge(data,mean_C,how='left',on=["Cliente"],copy=False)
    mean_P.drop_duplicates(subset=['Producto_ID',"Agencia_ID"],keep="first")
    data=pd.merge(data,mean_PA,how='left',on=['Producto_ID',"Agencia_ID"],copy=False)
    mean_P.drop_duplicates(subset=['Producto_ID',"Ruta_SAK"],keep="first")
    data=pd.merge(data,mean_PR,how='left',on=['Producto_ID',"Ruta_SAK"],copy=False)
    mean_P.drop_duplicates(subset=['Producto_ID',"Cliente_ID"],keep="first")
    data=pd.merge(data,mean_PC,how='left',on=['Producto_ID',"Cliente_ID"],copy=False)
    del mean_P,mean_C
    del mean_PA,mean_PR,mean_PC
    gc.collect()

    aux=data[['Producto_ID','meanP']]
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID'],keep="first")
    aux.drop(['index'],inplace=True,axis=1)
    test=pd.merge(test,aux,how='left',on=['Producto_ID'],copy=False)

    aux=data[['Cliente_ID','meanC']]
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Cliente_ID'],keep="first")
    aux.drop(['index'],inplace=True,axis=1)
    test=pd.merge(test,aux,how='left',on=['Cliente_ID'],copy=False)

    aux=data[['Producto_ID','Agencia_ID','meanPA']]
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID','Agencia_ID'],keep="first")
    aux.drop(['index'],inplace=True,axis=1)
    test=pd.merge(test,aux,how='left',on=['Producto_ID','Agencia_ID'],copy=False)

    aux=data[['Producto_ID','Ruta_SAK','meanPR']]
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID','Ruta_SAK'],keep="first")
    aux.drop(['index'],inplace=True,axis=1)
    test=pd.merge(test,aux,how='left',on=['Producto_ID','Ruta_SAK'],copy=False)

    aux=data[['Producto_ID','Cliente_ID','meanPC']]
    aux = aux.reset_index()
    aux=aux.drop_duplicates(subset=['Producto_ID','Cliente_ID'],keep="first")
    aux.drop(['index'],inplace=True,axis=1)
    test=pd.merge(test,aux,how='left',on=['Producto_ID','Cliente_ID'],copy=False)

    for k,v in summary.iteritems():
        del v
    data=pd.concat([data,test])
    #print (data.tail())
    verbose(data.info(memory_usage=True))
    verbose('Isolating weeks (again)')
    ids=data[data.tst==1]["id"].values
    data.drop(["tst","id"],inplace=True,axis=1)
    weeks=[]
    for i in range(3,12):
        weeks.append(data[data.Semana==i])
    #test_size=weeks[-3].shape[0]
    data=pd.concat(weeks)
    #print (data.tail(50))
    del data

    return weeks,ids
def predweeks(weeks,regressor,span=3):
    features=preproc_weeks2(weeks,10)

    predictions10=predict(features,regressor)
    print (predictions10)
    weeks[-2]["Demanda_uni_equil"]=predictions10
    features=preproc_weeks2(weeks,11)
    predictions20=predict(features,regressor)
    print (predictions20)
    return np.concatenate((predictions10, predictions20))

def writesubmision(ids,predictions,path="./data/submission.csv"):
    submision=pd.DataFrame({"Demanda_uni_equil":predictions,"id":ids.astype("int32")})
    submision=submision[['id','Demanda_uni_equil']]
    submision.to_csv(path,index=False)

def getraindata(weeks,span=3):
    weeks=pd.concat(weeks)
    verbose('Extracting span')
    data=preproc_weeks(weeks)
    print (data.head())
    labels=data["Demanda_uni_equil"].values
    data.drop(["Semana","Demanda_uni_equil"],inplace=True,axis=1)#Not usefull columns from here

    return data.as_matrix(),labels
def train(features,labels):
    T_train_xgb = xgb.DMatrix(features, labels)
    verbose ("Training...")
    params = {"objective":"reg:linear",
                               "booster" : "gbtree",
                               "eta":0.1,
                               "max_depth":10,
                               "subsample":0.85,
                               "colsample_bytree":0.7}
    regressor = xgb.train(dtrain=T_train_xgb,params=params)
    #regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    #regressor.fit(features, labels)

    return regressor
def predict(features,regressor):
    preds=regressor.predict(xgb.DMatrix(features))
    preds=np.exp(preds)-1
    #preds=regressor.predict(features)
    preds[preds<0]=0
    return preds

def train_test(features,labels,test_size):
    verbose ("Features size",features.shape)
    verbose ("Labels size",labels.shape)
    verbose ("Size test",test_size)
    T_train_xgb = xgb.DMatrix(features, labels)
    verbose ("Training...")
    params = {"objective":"reg:linear",
                                   "booster" : "gbtree",
                                   "eta":0.1,
                                   "max_depth":10,
                                   "subsample":0.85,
                                   "colsample_bytree":0.7}
    regressor = xgb.train(dtrain=T_train_xgb,params=params)
    verbose ("Training...")
    #regressor = skflow.TensorFlowLinearRegressor()#TODO convert uint32 to TensorFlow DType
    #regressor.fit(features[:-test_size], labels[:-test_size])
    verbose ("Predict...")
    preds=regressor.predict(xgb.DMatrix(features[-test_size:]))
    #preds=regressor.predict(features[-test_size:])
    preds[preds<0]=0
    verbose(preds)
    preds=np.exp(preds)-1
    verbose(len(preds))
    verbose("MSE: ")
    labels=np.exp(labels)-1
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
    p.add_argument("--size",default=None,type=int,
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
    df_test=temp(df_test,False)
    print (df_test.head(20))
    weeks,ids=prepare_train_data(df_train,df_test,size=opts.size)
    for i in weeks[:-2]:
        print (i.Semana.values)

    features,labels=getraindata(weeks[:-2])

    for i in weeks:
            print ("SHAAAPE :>>>>>>>",i.shape)

    print ("Test size >>>>>>>>>>>>")
    test_size=weeks[-3].shape[0]
    print (test_size)
    train_test(features,labels,test_size)
    regressor=train(features,labels)
    predictions=predweeks(weeks[-4:],regressor)
    print ("Pedicciones juntas: ")
    print (predictions)
    writesubmision(ids,predictions)
