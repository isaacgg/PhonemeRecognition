# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:17:55 2019

@author: isaac
"""
import tensorflow as tf
import os
import numpy as np

import Floyd.scripts.commons.utils as utils
from Floyd.scripts.commons.RNN_CTC_base import RNN_CTC_base

class BLSTM(RNN_CTC_base):
    def lstm_cell(self, n_hidden, dropout = 1):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True), 
                      input_keep_prob = dropout, output_keep_prob = dropout, state_keep_prob = dropout)

    def rnn_cell(self):        
        rnn_out, _, _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [self.lstm_cell(self.n_hidden, self.dropout_ph) for i in range(2)],
            [self.lstm_cell(self.n_hidden, self.dropout_ph) for i in range(2)],
            self.X,
            dtype=tf.float32,
            sequence_length=self.len_x)
        return rnn_out
    
    def logits_decode(self, logits, seq_length):
#        decoded, probs = tf.nn.ctc_beam_search_decoder(logits, seq_length, 100, 1)
#        return decoded[0],probs
        decoded, probs = tf.nn.ctc_greedy_decoder(logits, seq_length)
        return decoded[0], probs
    
    def optimizer_fn(self, lr, loss):
        return tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
        
    def lr_scheduler(self, x,e,d):
        return x
    
    def __init__(self, n_feats, n_hidden, n_classes, logsdir = None, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(n_feats = n_feats, n_classes = n_classes, logsdir = logsdir, **kwargs)

def normalize_data(X_train, X_test, folder_name):
    if os.path.isfile(folder_name + "mean.pkl"):
        mean = utils.load_pickle(folder_name + "mean.pkl")
        std = utils.load_pickle(folder_name + "std.pkl")
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

if __name__ == "__main__":
#    folder_name = "./PreprocessedData/MFCCs_delta1/hann/"
#    logdir = "./BLSTM/MFCC/Hann/"
    folder_name = "./PreprocessedData/RBMs_longtrain/hann/v2/400_50/"
    logdir = "./BLSTM/rbm_50_deltas_argmax/"
    
    epochs = 8
    lr = 0.001
    batchsize = 8
    dropout = 0.85
    
    n_feats = 102
    n_hidden = 128
    n_classes = 39 + 1
    
    X_train_path = folder_name + "X_train.npy"
    X_train = np.array(utils.load_npy(X_train_path))
    X_test_path = folder_name + "X_test.npy"
    X_test = np.array(utils.load_npy(X_test_path))
    
    y_train_path = folder_name + "y_train.npy"
    y_train = np.array(utils.load_npy(y_train_path))
    y_test_path = folder_name + "y_test.npy"
    y_test = np.array(utils.load_npy(y_test_path))

#    X_train,y_train = utils.get_mini_dataset(X_train,y_train,500)
#    X_test, y_test = utils.get_mini_dataset(X_test, y_test,100)

    X_train, X_test = normalize_data(X_train, X_test, folder_name)
    

    y_train = utils.remove_q(utils.to_ctc(np.array([utils.collapse_num_labels(y) for y in y_train])))
#    y_train = utils.to_ctc(np.array([utils.collapse_num_labels(y) for y in y_train]))

    y_test = utils.remove_q(utils.to_ctc(np.array([utils.collapse_num_labels(y) for y in y_test])))
#    y_test = utils.to_ctc(np.array([utils.collapse_num_labels(y) for y in y_test]))
    
    rnn = BLSTM(n_feats = n_feats, n_hidden = n_hidden, n_classes = n_classes,
                logsdir = logdir)
    
    rnn.set_data(X_train, X_test, y_train, y_test)

    rnn.fit(n_epochs=epochs,
            batch_size=batchsize,
            learning_rate=lr,
            dropout = dropout,
            shuffle=True,
            tboard = True,
            verbose=True,
            low_memory = False)

#    rnn.load_weights(logdir)
    
    rnn.fit(n_epochs=epochs,
            batch_size=batchsize,
            learning_rate=lr,
            dropout = 1,
            shuffle=True,
            tboard = True,
            verbose=True,
            low_memory = False)

    rnn.save_weights(logdir)

#for u in use_ctc:
#    for w in window:
#        for f in feats:
#            folder_name = "./data_checkpoints/data_ctc_" + w + "/"
#            logdir = "./" + "BLSTM" + "/" + f + "/" + w + "/ctc_" + str(u)
#            
#            X_train_path = folder_name + "X_train_" + f + ".pkl"
#            X_train = open_pickle(X_train_path)
#            X_test_path = folder_name + "X_test_" + f + ".pkl"
#            X_test = open_pickle(X_test_path)
#            
#            y_train_path = folder_name + "y_train.pkl"
#            y_train = open_pickle(y_train_path)
#            y_test_path = folder_name + "y_test.pkl"
#            y_test = open_pickle(y_test_path)
#            
#            X_train,y_train = get_mini_dataset(X_train,y_train,2000)
#            X_test, y_test = get_mini_dataset(X_test, y_test,500)
#            
#            get_mini_dataset
#            
#            n_feats = X_train[0].shape[1]
#            
#            rnn = RNN_base(n_feats = n_feats, n_hidden = 2*96, n_classes = 62, use_ctc = u, logs_dir = logdir, lr_scheduler=lr_scheduler)
#                        
#            rnn.fit(X_train,
#                    y_train,
#                    X_test,
#                    y_test,
#                    n_epoches=epochs,
#                    batch_size=4,
#                    learning_rate=lr,
#                    shuffle=True,
#                    tboard = True,
#                    verbose=True)                