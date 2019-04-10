# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:42:02 2019

@author: isaac
"""
import os
import numpy as np
import tensorflow as tf

import utils
from RNN_base import RNN_base

class BLSTM(RNN_base):
    _id = "BLSTM"
    def forward_pass(self): #Example implementation of forward_pass
        cellfw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.rnn_units, state_is_tuple=True), 
                                               input_keep_prob = self.dropout, output_keep_prob = self.dropout, state_keep_prob = self.dropout)
        
        cellbw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.rnn_units, state_is_tuple=True), 
                                               input_keep_prob = self.dropout, output_keep_prob = self.dropout, state_keep_prob = self.dropout)

        #This return [0 0 .. 0] for all timesteps higher than sequence_length 
        rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, self.X, dtype=tf.float32, sequence_length=self.len_x)        
        rnn_out = tf.concat(rnn_out,2)        
        
        
        return rnn_out    
    
    def __init__(self, n_feats = 26, n_hidden = 94, n_classes = 61, dropout = 1, use_ctc = True, logs_dir = None, **kwargs):
        self.dropout = dropout
        self.rnn_units = n_hidden
        RNN_base.__init__(self, n_feats = n_feats, n_hidden = n_hidden, n_classes = n_classes, use_ctc = use_ctc, logs_dir = logs_dir, **kwargs)


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

if __name__ == '__main__':
    
    logdir = "./LSTM_categories/"
    data_folder = "./PreprocessedData/RBMs_longtrain/hann/v1/400_600/"
    
    n_categories = 6 + 1
    
    #Params
    dropout = 0.7
    n_feats = 600
    n_hidden = 64
    
    #hyper-params
    lr = 0.001
    epochs = 80
    batchsize = 8
    use_ctc = True

    #Data extraction and preparation
    X_train_path = data_folder + "X_train.npy"
    X_train = np.array(utils.load_npy(X_train_path))
    
    X_test_path = data_folder + "X_test.npy"
    X_test = np.array(utils.load_npy(X_test_path))
    
    y_train_path = data_folder + "y_train.npy"
    y_train = utils.load_npy(y_train_path)
    y_train = np.array([utils.categories_to_int(utils.labels_num_to_category(utils.collapse_num_labels(y))) for y in y_train])
    
    y_test_path = data_folder + "y_test.npy"
    y_test = utils.load_npy(y_test_path)
    y_test = np.array([utils.categories_to_int(utils.labels_num_to_category(utils.collapse_num_labels(y))) for y in y_test])
    
    #SAVE MEMORY??
#    X_train, y_train = utils.get_mini_dataset(X_train, y_train, 500)
#    X_test, y_test = utils.get_mini_dataset(X_test, y_test, 100)
    
    X_train, X_test = normalize_data(X_train, X_test, data_folder)
    
    if use_ctc:
        y_train = utils.to_ctc(y_train)
        y_test = utils.to_ctc(y_test)
    
    #LSTM create and feed
    rnn = BLSTM(n_feats = n_feats, n_hidden = n_hidden, n_classes = n_categories,
                use_ctc = use_ctc, dropout = dropout, logs_dir = logdir, lr_scheduler= lambda x,y,z:x)
    
    rnn.set_data(X_train, X_test, y_train, y_test)
    del(X_train, X_test, y_train, y_test)
    
    rnn.load_weights(logdir, "weights.h5")
    
    rnn.fit(n_epoches=epochs,
            batch_size=batchsize,
            learning_rate=lr,
            shuffle=True,
            tboard = True,
            verbose=True)