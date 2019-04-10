# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 03:35:54 2018

@author: isaac
"""

import tensorflow as tf
from rbm import RBM


class GBRBM(RBM):
    def __init__(self, n_visible, n_hidden, sample_visible=False, sigma=1, l1 = 0.0, **kwargs):
        self.sample_visible = sample_visible
        self.sigma = sigma
        
        self.l1 = l1

        RBM.__init__(self, n_visible, n_hidden, **kwargs)

    def _initialize_vars(self):
        self.dropout_rate = tf.placeholder(tf.float32, shape = (), name = "dropout_rate") ###dropout

        
        hidden_mf = tf.matmul(self.x, self.w) + self.hidden_bias
        hidden_p = tf.nn.sigmoid(hidden_mf)
#        hidden_act = self.sample_sigmoid(hidden_mf, hidden_p)
        hidden_act = self.sample_bernoulli(hidden_p)
        
#        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias
        visible_recon_p = tf.matmul(hidden_act, tf.transpose(self.w)) + self.visible_bias
        visible_recon_act = self.sample_gaussian(visible_recon_p, self.sigma)
        
#        if self.sample_visible:
#            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_mf = tf.matmul(visible_recon_act, self.w) + self.hidden_bias
        hidden_recon_p = tf.nn.sigmoid(hidden_recon_mf)
#        hidden_recon_act = self.sample_sigmoid(hidden_recon_mf, hidden_recon_p)
        hidden_recon_act = self.sample_bernoulli(hidden_recon_p)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)/tf.to_float(tf.shape(self.x)[0])
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)/tf.to_float(tf.shape(self.x)[0])

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate_ph * x_new * (1 - self.momentum)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad) - self.learning_rate_ph*self.l1*tf.abs(self.w)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))/ tf.to_float(tf.shape(self.x)[0])
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))/ tf.to_float(tf.shape(self.x)[0])

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias
        
        self.features = self.compute_hidden        
        
        self.fix_hidden = self.hidden_bias.assign(0.1*(-1/2) + 0.1*1*(self.hidden_bias - tf.reduce_min(self.hidden_bias))/(tf.reduce_max(self.hidden_bias)-tf.reduce_min(self.hidden_bias)))

        
    def sample_sigmoid(self,x ,sigma):
        if sigma == 0:
            return tf.nn.sigmoid(x)
        return tf.nn.sigmoid(tf.random_normal(shape = tf.shape(x), mean = x, stddev = sigma))

    def sample_gaussian(self, x, sigma):
        return tf.random_normal(tf.shape(x), mean=x, stddev=sigma, dtype=tf.float32)

    def sample_bernoulli(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def get_features(self, batch_x):
        return self.sess.run(self.features, feed_dict={self.x: batch_x})
    
    def fix_hidden_bias(self):
        return self.sess.run(self.fix_hidden)

if __name__ == "__main__":
    
    import DataInfoExtractor as die
    from scipy import signal
    import os
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
#    plt.ion()

    winsize = 80
    n = 3
    losgdir = "D:/TrabajoFinDeMaster/gbRBM/"
    if not os.path.isdir(losgdir):
        os.makedirs(losgdir)
    window = signal.hann(winsize)
    
    ie = die.DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")    
    train_matrix = ie.get_n_window_per_phon_train(winsize, n)
    test_matrix = ie.get_n_window_per_phon_test(winsize, 1)

    mean = np.array(train_matrix).mean()#(axis = 0)
    std = np.array(train_matrix).std()#(axis = 0)
    with open(losgdir + "mean.pkl", 'wb') as handle:
        pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    with open(losgdir + "std.pkl", 'wb') as handle:
        pickle.dump(std, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    train_matrix = np.array([(r-mean)/(std) for r in train_matrix])
    test_matrix = np.array([(r-mean)/(std) for r in test_matrix])

    
    bbrbm = GBRBM(n_visible=80, n_hidden=120, learning_rate=0.01, momentum=0.5, use_tqdm=False)
    
    errs = bbrbm.fit(train_matrix, n_epoches=100, batch_size=50000)
    
    
    plt.plot(errs)
    plt.show()