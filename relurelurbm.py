# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:31:11 2018

@author: isaac
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 03:35:54 2018

@author: isaac
"""

import tensorflow as tf
from rbm import RBM
from util import sample_gaussian

class ReluReluRBM(RBM):
    def __init__(self, n_visible, n_hidden, sigma=1, l1 = 0.0, train_sigma = False, **kwargs):
        self.train_sigma = train_sigma
        if train_sigma:
            self.sigma = tf.Variable(sigma*tf.ones([n_visible]), dtype=tf.float32)
            self.delta_sigma = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        else:
            self.sigma = sigma

        self.l1 = l1
        self.lr_penalty = 1

        RBM.__init__(self, n_visible, n_hidden, **kwargs)

    def _initialize_vars(self):   
        self.dropout_rate = tf.placeholder(tf.float32, shape = (), name = "dropout_rate") ###dropout
        weigths_dropout = tf.nn.dropout(self.w, self.dropout_rate)###dropout
        
        hidden_mf = tf.matmul(self.x, self.w) + self.hidden_bias
        hidden_p = tf.nn.sigmoid(hidden_mf)
        hidden_act = self.sample_relu(hidden_mf, hidden_p)
        
        visible_recon_mf = tf.matmul(hidden_act, tf.transpose(self.w)) + self.visible_bias
        visible_recon_p = tf.nn.sigmoid(visible_recon_mf)
        visible_recon_act = self.sample_relu(visible_recon_mf, visible_recon_p)

        hidden_recon_mf = tf.matmul(visible_recon_act, weigths_dropout) + self.hidden_bias
        hidden_recon_p = tf.nn.sigmoid(hidden_recon_mf)
        hidden_recon_act = self.sample_relu(hidden_recon_mf, hidden_recon_p)

#        dropout_cte = tf.nn.dropout(tf.ones_like(hidden_mf), self.dropout_rate)###dropout
#        hidden_recon_act = tf.multiply(hidden_recon_act, dropout_cte)###dropout

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_act)/tf.to_float(tf.shape(self.x)[0])
        negative_grad = tf.matmul(tf.transpose(visible_recon_act), hidden_recon_act)/tf.to_float(tf.shape(self.x)[0])

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate_ph * x_new * (1 - self.momentum)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad) - self.learning_rate_ph*self.l1*tf.abs(self.w)
#        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_act, 0))/128 #+ self.learning_rate_ph*self.l1*tf.abs(self.visible_bias)
#        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))/128 #+ self.learning_rate_ph*self.l1*tf.abs(self.hidden_bias)

        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_act, 0)) #+ self.learning_rate_ph*self.l1*tf.abs(self.visible_bias)
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_act - hidden_recon_act, 0)) # + self.learning_rate_ph*self.l1*tf.abs(self.hidden_bias)
                                                                        #hidden_act
        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        if self.train_sigma:
            delta_sigma_new = 0.01*f(self.delta_sigma, 
                                     tf.reduce_mean((self.x**2  - visible_recon_p**2 + 2*self.visible_bias*(visible_recon_p - self.x))/(self.sigma), 0, False))       

            update_delta_sigma = self.delta_sigma.assign(delta_sigma_new)
            update_sigma = self.sigma.assign(self.sigma + delta_sigma_new)

            self.update_deltas += [update_delta_sigma]
            self.update_weights += [update_sigma]

        self.compute_hidden = self.sample_relu(tf.matmul(self.x, self.w) + self.hidden_bias, 0)
        self.compute_visible = self.sample_relu(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias, 0)
        self.compute_visible_from_hidden = self.sample_relu(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias, 0)

        self.features = self.compute_hidden
        self.featuresv2 = tf.nn.sigmoid(tf.matmul(self.x,self.w) + self.hidden_bias)
        self.featuresv3 = tf.matmul(self.x, self.w) + self.hidden_bias
        self.featuresv4 = tf.matmul(self.x, self.w)

#    def sample_gaussian(self, x, sigma):
#        return tf.random_normal(tf.shape(x), mean=x, stddev=sigma, dtype=tf.float32)

    def sample_relu(self, x, sigma):
        if sigma == 0:
            return tf.nn.relu(x)
        return tf.nn.relu(tf.random_normal(shape = tf.shape(x), mean = x, stddev = sigma))

    def get_features(self, batch_x):
        return self.sess.run(self.features, feed_dict={self.x: batch_x})

    def get_features_v2(self, batch_x):
        return self.sess.run(self.featuresv2, feed_dict={self.x: batch_x})
    
    def get_features_v3(self, batch_x):
        return self.sess.run(self.featuresv3, feed_dict={self.x: batch_x})

    def get_features_v4(self, batch_x):
        return self.sess.run(self.featuresv4, feed_dict={self.x: batch_x})


if __name__ == "__main__":
    import DataInfoExtractor as die
    from scipy import signal
    import os
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from utils import plot_weights, plot_input_bias, plot_hidden_bias

    winsize = 400
    losgdir = "D:/TrabajoFinDeMaster/greluRBM_400_600_hann/"
    
    if False:
        if not os.path.isdir(losgdir):
            os.makedirs(losgdir)
        window = signal.hann(winsize)#np.ones(winsize)
        
        ie = die.DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")
        
        train_matrix = ie.get_train_matrix_from_wavs(400, 160)
#        test_matrix = ie.get_test_matrix_from_wavs(400, 160)

        mean = np.array(train_matrix).mean()#(axis = 0)
        std = np.array(train_matrix).std()#(axis = 0)
        with open(losgdir + "mean.pkl", 'wb') as handle:
            pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        with open(losgdir + "std.pkl", 'wb') as handle:
            pickle.dump(std, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
        train_matrix = np.array([window*(r-mean)/(std) for r in train_matrix])
#        test_matrix = np.array([window*(r-mean)/(std) for r in test_matrix])
        
    epochs = 50
    batchsize = 128
    """#ESTO ESTA BAJO, IGUAL HAY QUE SUBRILO"""
    lr = 0.005
    def lr_scheduler(x,e,d, epochs = epochs, lr_init = lr):
        if e != 0 and e != (epochs-1):
            x *= (1-(e/epochs)**8)/((1-((e-1)/epochs)**8)) 
        else:
            (1-(e/epochs)**8)
        if d<0:
            x /= 2
        return x

    grelurbm = GReluRBM(n_visible=400, n_hidden=600, momentum = 0, l1 = 0.003, use_tqdm=False, err_function = 'rmse')
    errs = grelurbm.fit(train_matrix, n_epoches=epochs, batch_size=batchsize, learning_rate=lr, lr_scheduler = lr_scheduler)
    
    plt.plot(errs)
    plt.show()

    plot_weights(grelurbm, 50)
    plot_input_bias(grelurbm)
    plot_hidden_bias(grelurbm)
    
    grelurbm.save_weights(losgdir, "weights.h5")
    
    #TODO: Así aprende las características dpm
#        positive_grad = tf.matmul(tf.transpose(self.x), hidden_act)/tf.to_float(tf.shape(self.x)[0])
#        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_act)/tf.to_float(tf.shape(self.x)[0])
#
#        def f(x_old, x_new):
#            return self.momentum * x_old +\
#                   self.learning_rate * x_new * (1 - self.momentum)
#
#        delta_w_new = f(self.delta_w, positive_grad - negative_grad) - self.learning_rate*self.l1*tf.abs(self.w)
#        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0)) #- self.learning_rate*self.l1*tf.abs(self.visible_bias)
#        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_mf - hidden_recon_mf, 0)) #- self.learning_rate*self.l1*tf.abs(self.hidden_bias)