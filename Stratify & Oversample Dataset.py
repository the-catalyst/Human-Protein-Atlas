import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

def oversample_df(X_train, df):
    multi_rare = np.array([1,1,1,1,1,1,1,
                  1,4,4,4,1,1,1,
                  1,4,1,1,1,1,2,
                  1,1,1,1,1,1,4])
    
    os_df = df.copy()
    df.set_index('Id', inplace=True)
    df.Target = [[int(i) for i in t.split()] for t in df.Target]
    
    for protein_id in X_train:
        labels = df.loc[protein_id,'Target']
        
        copies = max(multi_rare[tuple(labels)])
        if copies > 1:
            row = os_df[os_df.Id == protein_id[0]]
            for c in range(copies-1):
                os_df = os_df.append(row, ignore_index=True)
        
    return os_df

def create_labellist(df):
    y = np.array([list(map(int, t.split(' '))) for t in df.Target])
    
    Label_list = np.zeros((len(y), 28), dtype = int)
    
    for i, k in enumerate(y):
        Label_list[i, k] = 1
    
    return Label_list

def stratified_split(df):
    X = np.array([x for x in df.Id])[:,None]
    Y = create_labellist(df)
    
    np.random.seed(42)
    return iterative_train_test_split(X, Y, 0.15)

def create_dataset(df):
    X_train,_,X_valid,_ = stratified_split(df)
    
    is_valid = [False]*len(df)
    for i, ID in enumerate(list(df.Id)):
        if ID in X_valid:
            is_valid[i] = True
    df.insert(2, 'is_valid', is_valid)
    df = oversample_df(X_train, df)
    return df