# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:58:37 2019

@author: isaac
"""
import numpy as np
import tensorflow as tf

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
#        return decoded[0], probs
        decoded, probs = tf.nn.ctc_greedy_decoder(logits, seq_length)
        return decoded[0], probs
    
    def optimizer_fn(self, lr, loss):
        return tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
        
    def lr_scheduler(self, x,e,d):
        return x
    
    def __init__(self, n_feats, n_hidden, n_classes, logsdir = None, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(n_feats = n_feats, n_classes = n_classes, logsdir = logsdir, **kwargs)
        
    def zero_pad(self, X):
        len_x = np.array([len(x) for x in X])
        
        if X[0].ndim == 2:
            X_cpy = np.zeros((len(X), np.max(len_x), self.n_feats))
            for i,j in enumerate(X):
                X_cpy[i,:len(j),:] = j
            del(X)
        else:
            X_cpy = -1*np.ones((len(X), np.max(len_x)))
            for i,j in enumerate(X):
                X_cpy[i,:len(j)] = j
            del(X)

        return X_cpy, len_x
      
    def custom_accuracy_voting3(self, X_test, y_test, batch_size = 128, window_offset = 5):
        accuracies = []
        
        n_data_test = X_test.shape[0]
        
        index_to_compare = int((utils.to_ctc(y_test).shape[1]-1)/2)

        if batch_size > 0:
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_test = 1
            
        X_test_cpy, len_x_test = self.zero_pad(X_test)
        y_test_cpy, len_y_test = self.zero_pad(y_test)
        del(X_test, y_test)
        
        for b in range(n_batches_test):
            batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]

            y_true = [y[y!=-1] for y in batch_y] #[np.trim_zeros(y, trim='b') for y in batch_y]
            valid_phon = []
            real_batch_x = []
            real_batch_y = []
            for y_t, b_x, b_y in zip(y_true, batch_x, batch_y):
                y_t_ctc = utils.to_ctc([y_t])[0]
                if len(y_t_ctc) != 5:
                    print("Stop")
                label_to_compare = y_t_ctc[index_to_compare]
                l_label = y_t_ctc[index_to_compare - 1]
                r_label = y_t_ctc[index_to_compare + 1]
                
                ixs_true = np.where((y_t == label_to_compare) | (y_t == r_label) | (y_t == l_label))[0]
                
                if np.max(ixs_true) + window_offset + 1 > len(y_t):
                    offset_right = np.arange(np.max(ixs_true) + 1, len(y_t))
                else:
                    offset_right = np.arange(np.max(ixs_true) + 1, np.max(ixs_true) + window_offset + 1)
                    
                if np.min(ixs_true) - window_offset < 0:
                    offset_left = np.arange(0, np.min(ixs_true))
                else:
                    offset_left = np.arange(np.min(ixs_true) - window_offset, np.min(ixs_true))
                final_ix = offset_left.tolist() + ixs_true.tolist() + offset_right.tolist()
                valid_phon.append(label_to_compare)
                real_batch_x.append(b_x[final_ix])
                real_batch_y.append(y_t[final_ix])
            
            real_batch_x, real_batch_len = self.zero_pad(real_batch_x)
                
            y_pred = self._session.run(self.framewise_output, feed_dict = {self.X: np.array(real_batch_x), 
                                                                           self.len_x: np.array(real_batch_len), 
                                                                           self.dropout_ph: 1})
            y_pred = np.transpose(y_pred)
            
            for y_p, y_t, label in zip(y_pred, real_batch_y, valid_phon):
                ixs_true = np.where(y_t == label)[0]
                ixs_pred = np.where(y_p != 40)[0]
                ixs_to_compare = list(set(ixs_true).intersection(ixs_pred))

                if len(ixs_to_compare) == 0:
                    acc = 0
                    accuracies.append(acc)
                    continue
                
                array_compare = np.array(y_t)[ixs_to_compare] - np.array(y_p)[ixs_to_compare]
                acc = (np.count_nonzero(array_compare==0))/len(np.array(y_t)[ixs_to_compare])
                
                accuracies.append(acc)            

        return accuracies    
    
    def custom_accuracy_voting2(self, X_test, y_test, batch_size = 128, window_offset = 5):
        accuracies = []
        
        n_data_test = X_test.shape[0]
        
        index_to_compare = int((utils.to_ctc(y_test).shape[1]-1)/2)

        if batch_size > 0:
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_test = 1
            
        X_test_cpy, len_x_test = self.zero_pad(X_test)
        y_test_cpy, len_y_test = self.zero_pad(y_test)
        del(X_test, y_test)
        
        for b in range(n_batches_test):
            batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]

            y_true = [y[y!=-1] for y in batch_y] #[np.trim_zeros(y, trim='b') for y in batch_y]
            valid_phon = []
            real_batch_x = []
            real_batch_y = []
            for y_t, b_x, b_y in zip(y_true, batch_x, batch_y):
                y_t_ctc = utils.to_ctc([y_t])[0]
                label_to_compare = y_t_ctc[index_to_compare]
                ixs_true = np.where(y_t == label_to_compare)[0]
                
                if np.max(ixs_true) + window_offset + 1 > len(y_t):
                    offset_right = np.arange(np.max(ixs_true) + 1, len(y_t))
                else:
                    offset_right = np.arange(np.max(ixs_true) + 1, np.max(ixs_true) + window_offset + 1)
                    
                if np.min(ixs_true) - window_offset < 0:
                    offset_left = np.arange(0, np.min(ixs_true))
                else:
                    offset_left = np.arange(np.min(ixs_true) - window_offset, np.min(ixs_true))
                final_ix = offset_left.tolist() + ixs_true.tolist() + offset_right.tolist()
                valid_phon.append(label_to_compare)
                real_batch_x.append(b_x[final_ix])
                real_batch_y.append(y_t[final_ix])
            
            real_batch_x, real_batch_len = self.zero_pad(real_batch_x)
                
            y_pred = self._session.run(self.framewise_output, feed_dict = {self.X: np.array(real_batch_x), 
                                                                           self.len_x: np.array(real_batch_len), 
                                                                           self.dropout_ph: 1})
            y_pred = np.transpose(y_pred)
            
            for y_p, y_t, label in zip(y_pred, real_batch_y, valid_phon):
                ixs_true = np.where(y_t == label)[0]
                ixs_pred = np.where(y_p != 40)[0]
                ixs_to_compare = list(set(ixs_true).intersection(ixs_pred))

                if len(ixs_to_compare) == 0:
                    acc = 0
                    accuracies.append(acc)
                    continue
                
                array_compare = np.array(y_t)[ixs_to_compare] - np.array(y_p)[ixs_to_compare]
                acc = (np.count_nonzero(array_compare==0))/len(np.array(y_t)[ixs_to_compare])
                
                accuracies.append(acc)            

        return accuracies
        
    def custom_accuracy_voting(self, X_test, y_test, batch_size = 128, index_to_compare = 1):
        accuracies = []
        
        n_data_test = X_test.shape[0]

        if batch_size > 0:
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_test = 1
            
        len_x_test = np.array([len(x) for x in X_test])
        len_y_test = np.array([len(y) for y in y_test])
 
        X_test_cpy = np.zeros((len(X_test), np.max(len_x_test), self.n_feats))
        for i,j in enumerate(X_test):
            X_test_cpy[i,:len(j),:] = j
        del(X_test)

        y_test_cpy = np.zeros((len(y_test), np.max(len_y_test)))
        for i,j in enumerate(y_test):
            y_test_cpy[i,:len(j)] = j
        del(y_test)

        for b in range(n_batches_test):
            batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_len_x = len_x_test[b * batch_size:(b + 1) * batch_size]
                        
            y_pred = self._session.run(self.framewise_output, feed_dict = {self.X: batch_x, 
                                                                  self.len_x: batch_len_x,
                                                                  self.dropout_ph: 1})
            y_pred = np.transpose(y_pred)
            y_true = [y[y!=-1] for y in batch_y] #[np.trim_zeros(y, trim='b') for y in batch_y]
                    
            for y_t, y_p in zip(y_true, y_pred):            
                y_t_ctc = utils.to_ctc([y_t])[0]
                if len(y_t_ctc) <= index_to_compare: 
                    acc = 0
                else:
                    label_to_compare = y_t_ctc[index_to_compare]
                    ixs_true = np.where(y_t == label_to_compare)[0]
                    ixs_pred = np.where(y_p != 40)[0]
                    
                    ixs_to_compare = list(set(ixs_true).intersection(ixs_pred))
                    if len(ixs_to_compare) == 0:
                        acc = 0
                    else:
                        array_compare = np.array(y_t)[ixs_to_compare] - np.array(y_p)[ixs_to_compare]
                        acc = (np.count_nonzero(array_compare==0))/len(np.array(y_t)[ixs_to_compare] )
                    
                accuracies.append(acc)
            
        return accuracies
    
    def custom_accuracy(self, X_test, y_test, batch_size = 128, index_to_compare = 1):
        accuracies = []
        
        n_data_test = X_test.shape[0]

        if batch_size > 0:
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_test = 1
            
        len_x_test = np.array([len(x) for x in X_test])
        len_y_test = np.array([len(y) for y in y_test])
 
        X_test_cpy = np.zeros((len(X_test), np.max(len_x_test), self.n_feats))
        for i,j in enumerate(X_test):
            X_test_cpy[i,:len(j),:] = j
        del(X_test)

        y_test_cpy = np.zeros((len(y_test), np.max(len_y_test)))
        for i,j in enumerate(y_test):
            y_test_cpy[i,:len(j)] = j
        del(y_test)

        for b in range(n_batches_test):
            batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]
            batch_len_x = len_x_test[b * batch_size:(b + 1) * batch_size]
            batch_len_y = len_y_test[b * batch_size:(b + 1) * batch_size]        
                        
            y_pred = self._session.run(self.framewise_output, feed_dict = {self.X: batch_x, 
                                                                  self.len_x: batch_len_x,
                                                                  self.dropout_ph: 1})
            
            #Create framewise y_true and y_pred        
            y_true = [np.trim_zeros(y) for y in batch_y]
            y_predicted = []
            for i, ngram in enumerate(np.transpose(y_pred)):
                phons_ix = np.where(ngram != 40)[0]
                
                labels = -1 * np.ones(batch_len_y[i])
                if len(phons_ix) > 0:
                    labels[:phons_ix[0]] = [ngram[phons_ix[0]]]*phons_ix[0]
                    for ind, phon_ix in enumerate(phons_ix):
                        letter = ngram[phon_ix]
                        if ind == len(phons_ix)-1:
                            labels[phon_ix:] = [letter]*(len(labels)-phon_ix)
                        else:
                            labels[phon_ix:phons_ix[ind + 1]] = [letter]*(phons_ix[ind + 1] - phon_ix)
                
                y_predicted.append(labels)

            #Compare the index that matters
            for y_t, y_p in zip(y_true, y_predicted):
                y_t_ctc = utils.to_ctc([y_t])[0]
                if len(y_t_ctc) <= index_to_compare: 
                    continue
                
                label_to_compare = y_t_ctc[index_to_compare]
                ixs = np.where(y_t == label_to_compare)[0]
                
                array_compare = np.array(y_t)[ixs] - np.array(y_p)[ixs]
                acc = (np.count_nonzero(array_compare==0))/len(np.array(y_t)[ixs] )
                
                accuracies.append(acc)
                
        return accuracies
                

if __name__ == '__main__':
    ngram = 5
    
    rootdir = "./LSTM_ngram/"    
    checkpointfolder = rootdir + "checkpoints/"
    checkpointfolder_specific = checkpointfolder + str(ngram) + "gram_v2/"
    logdir = "./Floyd/scripts/LSTM_phons/" #rootdir + "/logs/" + str(ngram) + "gram/"
        
    n_categories = 40 + 1
    
    #Params
    dropout = 0.5
    n_feats = 600
    n_hidden = 128
    
    #hyper-params
    lr = 0.001
    epochs = 80
    batchsize = 8
    use_ctc = True

            
    print("Load data")
    X_test = np.array(utils.load_npy(checkpointfolder_specific + "X_test_0.npy"))
    y_test = np.array(utils.load_npy(checkpointfolder_specific + "y_test_0.npy"))

#    X_train, y_train = utils.get_mini_dataset(X_train, y_train, 7000)
    X_test, y_test = utils.get_mini_dataset(X_test, y_test, 3000)

    #LSTM create and feed
    rnn = BLSTM(n_feats = n_feats, n_hidden = n_hidden, n_classes = n_categories,
                logsdir = logdir)
    
    rnn.load_weights(logdir)
    
    if ngram == 3:
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 5)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 4)
        print(np.mean(accuracy0))    
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 3)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 2)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 1)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, window_offset = 0)
        print(np.mean(accuracy0))

    if ngram == 5:
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 5)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 4)
        print(np.mean(accuracy0))    
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 3)
        print(np.mean(accuracy0))    
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 2)
        print(np.mean(accuracy0))    
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 1)
        print(np.mean(accuracy0))
        accuracy0 = rnn.custom_accuracy_voting3(X_test, y_test, batch_size = 128, window_offset = 0)
        print(np.mean(accuracy0))
    
    plt.plot([0.4799777777777777,
0.47758333333333336,
0.47436666666666666,
0.46683888888888886,
0.45885,
0.44809444444444446])

    plt.plot(
[0.4335,
0.4251666666666667,
0.41733333333333333,
0.3998333333333333,
0.3598611111111111,
0.32065])
    plt.show()
    
#    accuracy1 = rnn.custom_accuracy_voting2(X_test, y_test, batch_size = 128, index_to_compare = 1)
    