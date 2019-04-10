# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 01:17:49 2019

@author: isaac
"""
import pandas as pd
import numpy as np
import os

import Floyd.scripts.commons.utils as utils

def create_phon_df(X,Y):
    labels_df = pd.DataFrame(columns=["ix", "phon", "start", "stop", "feats"])
    for ix, y in enumerate(Y):
        feats = []
        start = 0
        x = X[ix] #para cada frase
        for i in range(len(y)-1): #para cada frame (menos el útlimo), mientras sea el mismo fonema extendemos feats, cuando cambie lo guardamos
            if y[i]!=y[i+1]: 
                feats.append(x[i,:])
                stop = i
                labels_df = labels_df.append({"ix":ix, "phon": y[i], "start": start, "stop": stop, "feats": np.array(feats)}, ignore_index = True)
                start = i+1
                feats = []
            else:
                feats.append(x[i,:])    
    return labels_df

def create_ngram(df, n):
    X = []
    y = []
    for d in df.groupby("ix"):
        for i in range(len(d[1])-n):
            try:
                X.append(np.concatenate(d[1].iloc[i:i+n]["feats"].values))
                y.append(d[1].iloc[i:i+n]["phon"].values)
            except ValueError:
                print("Problema en ix: " + str(d[1].iloc[i]["ix"]) + ", i: " + str(i))
    return X,y

def create_ngram_v2(df, n):
    X = []
    y = []
    for phrase in df.groupby("ix"):
        d = phrase[1].sort_values("start")
        for i in range(len(d)-n):
            try:
                feats = np.concatenate(d.iloc[i:i+n]["feats"].values)
                labels = d.iloc[i:i+n].apply(lambda row: np.repeat(row["phon"], row["stop"] - row["start"] + 1), axis = 1).values
                labels = np.concatenate(labels)
                
                assert len(labels) == len(feats)
                
                X.append(feats)
                y.append(labels)
            except ValueError:
                print("Problema en ix: " + str(d.iloc[i]["ix"]) + ", i: " + str(i))
    return X,y

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def normalize_data(X_train, X_test, folder_name):
    if os.path.isfile(folder_name + "mean.pkl"):
        mean = utils.open_pickle(folder_name + "mean.pkl")
        std = utils.open_pickle(folder_name + "std.pkl")
    else:
        train_matrix = []
        for x in X_train:
            train_matrix.extend(x)
        train_matrix = np.array(train_matrix)
        
        mean = np.array(train_matrix).mean(0)
        std = np.array(train_matrix).std(0)
        
        utils.save_pickle(folder_name + "mean.pkl", mean)
        utils.save_pickle(folder_name + "std.pkl", std)
    
    X_train = [(x-mean)/std for x in X_train]
    X_test = [(x-mean)/std for x in X_test]
    
    return np.array(X_train), np.array(X_test)
        
def create_dfs(data_folder, checkpointfolder):
    if os.path.isfile(checkpointfolder + "train_df.pkl"):
        assert os.path.isfile(checkpointfolder + "test_df.pkl")
        print("dfs already exist")
        
        return
    else:
        print("Creating dfs")
        #Data extraction and preparation
        X_train_path = data_folder + "X_train.npy"
        X_train = np.array(utils.load_npy(X_train_path))
        
        X_test_path = data_folder + "X_test.npy"
        X_test = np.array(utils.load_npy(X_test_path))
        
        y_train_path = data_folder + "y_train.npy"
        y_train = utils.load_npy(y_train_path)
        y_train = np.array([utils.collapse_num_labels(y) for y in y_train])
        
        y_test_path = data_folder + "y_test.npy"
        y_test = utils.load_npy(y_test_path)
        y_test = np.array([utils.collapse_num_labels(y) for y in y_test])
        
        #SAVE MEMORY??
        X_train, X_test = normalize_data(X_train, X_test, data_folder)
        
        train_df = create_phon_df(X_train, y_train)
        del(X_train, y_train)
        train_df.to_pickle(checkpointfolder + "train_df.pkl")
        
        test_df = create_phon_df(X_test, y_test)
        del(X_test, y_test)
        test_df.to_pickle(checkpointfolder + "test_df.pkl")


        
if __name__ == "__main__":
    ngram = 7
    
    rootdir = "./LSTM_ngram/"    
    checkpointfolder = rootdir + "checkpoints/"
    checkpointfolder_specific = checkpointfolder + str(ngram) + "gram_v2/"
    makedir(checkpointfolder_specific)

    data_folder = "./PreprocessedData/RBMs_longtrain/hann/v2/400_600/"

    create_dfs(data_folder, checkpointfolder)
            
    if os.path.isfile(checkpointfolder + "train_df.pkl"):
        assert os.path.isfile(checkpointfolder + "test_df.pkl")
    
        print("Create .npys")
        train_df = pd.read_pickle(checkpointfolder + "train_df.pkl")
        X_train, y_train = create_ngram_v2(train_df, ngram)
        del(train_df)
        
        print("save train")
        utils.save_npy(y_train[:len(y_train)//5], checkpointfolder_specific + "y_train_0.npy")
        utils.save_npy(y_train[len(y_train)//5:2*len(y_train)//5], checkpointfolder_specific + "y_train_1.npy")
        utils.save_npy(y_train[2*len(y_train)//5:3*len(y_train)//5], checkpointfolder_specific + "y_train_2.npy")
        utils.save_npy(y_train[3*len(y_train)//5:4*len(y_train)//5], checkpointfolder_specific + "y_train_3.npy")
        utils.save_npy(y_train[4*len(y_train)//5:], checkpointfolder_specific + "y_train_4.npy")

        del(y_train)
        utils.save_npy(X_train[:len(X_train)//5], checkpointfolder_specific + "X_train_0.npy")
        utils.save_npy(X_train[len(X_train)//5:2*len(X_train)//5], checkpointfolder_specific + "X_train_1.npy")
        utils.save_npy(X_train[2*len(X_train)//5:3*len(X_train)//5], checkpointfolder_specific + "X_train_2.npy")
        utils.save_npy(X_train[3*len(X_train)//5:4*len(X_train)//5], checkpointfolder_specific + "X_train_3.npy")
        utils.save_npy(X_train[4*len(X_train)//5:], checkpointfolder_specific + "X_train_4.npy")
        del(X_train)

        print("Create .npys")
        test_df = pd.read_pickle(checkpointfolder + "test_df.pkl")        
        X_test, y_test = create_ngram_v2(test_df, ngram)
        del(test_df)
        
        print("save test")
        utils.save_npy(y_test[:len(y_test)//5], checkpointfolder_specific + "y_test_0.npy")
        utils.save_npy(y_test[len(y_test)//5:2*len(y_test)//5], checkpointfolder_specific + "y_test_1.npy")
        utils.save_npy(y_test[2*len(y_test)//5:3*len(y_test)//5], checkpointfolder_specific + "y_test_2.npy")
        utils.save_npy(y_test[3*len(y_test)//5:4*len(y_test)//5], checkpointfolder_specific + "y_test_3.npy")
        utils.save_npy(y_test[4*len(y_test)//5:], checkpointfolder_specific + "y_test_4.npy")
        del(y_test)

        utils.save_npy(X_test[:len(X_test)//5], checkpointfolder_specific + "X_test_0.npy")
        utils.save_npy(X_test[len(X_test)//5:2*len(X_test)//5], checkpointfolder_specific + "X_test_1.npy")
        utils.save_npy(X_test[2*len(X_test)//5:3*len(X_test)//5], checkpointfolder_specific + "X_test_2.npy")
        utils.save_npy(X_test[3*len(X_test)//5:4*len(X_test)//5], checkpointfolder_specific + "X_test_3.npy")
        utils.save_npy(X_test[4*len(X_test)//5:], checkpointfolder_specific + "X_test_4.npy")
        del(X_test)