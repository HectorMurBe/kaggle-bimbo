#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function  # Python 2 users only
import argparse
import pandas as pd
import numpy as np

train_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 
               'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}

test_types = {'Agencia_ID':np.uint16, 'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 
              'Producto_ID':np.uint16, 'id':np.uint16}




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
    df_train = pd.read_csv(opts.train, usecols=train_types.keys(), dtype=train_types)
    df_test = pd.read_csv(opts.test,usecols=test_types.keys(), dtype=test_types)

    print("Columns for train")
    print(df_train.size)

    print("Columns for test")
    print(df_test.size)
