# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 03:36:35 2018

@author: isaac
"""

import tensorflow as tf
import numpy as np
import sys
from util import tf_xavier_init


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 logsdir = None,
                 # DEPRECATED:
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine', 'rmse'}:
            raise ValueError('err_function should be either \'mse\', \'rmse\' or \'cosine\'')
    
        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate_ph = tf.placeholder(tf.float32, ())#learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])
        
#        self.w = tf.Variable(tf.random_normal(mean=0.0, stddev=(1/self.n_visible), dtype = tf.float32, shape = [self.n_visible, self.n_hidden]))
#        self.w = tf.Variable(tf.random_normal(mean=0.0, stddev=(0.1), dtype = tf.float32, shape = [self.n_visible, self.n_hidden]))
        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        
        
        self.visible_bias = tf.Variable(1*tf.ones([self.n_visible]), dtype=tf.float32)
#        self.hidden_bias = tf.Variable(tf.random_uniform(minval = 0, maxval = 1, dtype = tf.float32, shape = [self.n_hidden]))
        self.hidden_bias = tf.Variable(1*tf.ones([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
    
        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        elif err_function == 'mse':
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))
        else: #rmse
            self.compute_err = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.compute_visible)))


        hist_bv = tf.summary.histogram('visible_bias', self.visible_bias)        
        hist_b = tf.summary.histogram('hidden_bias', self.hidden_bias)
        hist_w = tf.summary.histogram("w", self.w)
                                
        self.hists_vars = tf.summary.merge([hist_w] + [hist_b] + [hist_bv])
        
        self.loss_ph = tf.placeholder(tf.float32, shape = (), name="loss_placeholder")
        self.summary_loss = tf.summary.scalar("loss", self.loss_ph)
        
        self.hist_metrics = tf.summary.merge([self.summary_loss])


        self.logs_dir = logsdir

        if self.logs_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.logs_dir+"/logs/train")
#            self.test_writer = tf.summary.FileWriter(self.logs_dir+"/logs/test")


        self.hidden_bias_below_zero = tf.cast(tf.math.greater(tf.zeros_like(self.hidden_bias), self.hidden_bias), tf.float32)
        self.bias_below_zero = tf.cast(self.hidden_bias.assign(tf.abs(self.hidden_bias)), tf.float32)
        self.hidden_bias_correction = self.hidden_bias.assign(self.hidden_bias*self.hidden_bias_below_zero)

#        self.hidden_bias_below_zero = tf.cast(tf.math.greater(tf.zeros_like(self.hidden_bias), self.hidden_bias), tf.float32)
#        self.hidden_bias_correction = self.hidden_bias.assign_add(tf.abs(tf.reduce_mean(self.hidden_bias))*self.hidden_bias_below_zero)

        self.visible_bias_below_zero = tf.cast(tf.math.greater(tf.zeros_like(self.visible_bias), self.visible_bias), tf.float32)
        self.visible_bias_correction = self.visible_bias.assign_add(tf.abs(tf.reduce_mean(self.visible_bias))*self.visible_bias_below_zero)
        
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def test(self):
        return self.sess.run([self._test, self.hidden_bias])

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x, self.dropout_rate: 0})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x, dropout):
        self.sess.run(self.update_weights, feed_dict={self.x: batch_x, self.learning_rate_ph: self.learning_rate, self.dropout_rate: dropout})

    def write_train_metrics(self, loss, e):
        metrics, hists = self.sess.run([self.hist_metrics, self.hists_vars], feed_dict={self.loss_ph: loss})
        self.train_writer.add_summary(metrics, e)
        self.train_writer.add_summary(hists, e)
#        self.train_writer.flush()
        
    def correct_hidden_biases(self):
        self.sess.run(self.hidden_bias_correction)

    def correct_visible_biases(self):
        self.sess.run(self.visible_bias_correction)


    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            learning_rate=0.01,
            dropout = 0,
            lr_scheduler = lambda x,e,err:x,
            shuffle=False,
            tboard = True,
            correct_hidden = False,
            correct_visible = False,
            verbose=True):
        
        self.lr_scheduler = lr_scheduler
        
        assert n_epoches > 0

        self.learning_rate = learning_rate
        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        p_err_mean = np.inf
        for e in range(n_epoches):
            print("lr: " + str(self.learning_rate))
            
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)
            
            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x, dropout = dropout)
                if correct_hidden:
                    self.correct_hidden_biases()                                                                   #BIAS CORRECTION
                if correct_visible:
                    self.correct_visible_biases()
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if tboard:
                if self.logs_dir is not None:
                    self.write_train_metrics(np.mean(epoch_errs), e)

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()
            
            delta_err = p_err_mean - err_mean
            p_err_mean = err_mean
            self.learning_rate = self.lr_scheduler(self.learning_rate, e, delta_err)
            
            errs = np.hstack([errs, epoch_errs])            
        return errs
    
    def close_session(self):
        self.sess.close()
        
    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)