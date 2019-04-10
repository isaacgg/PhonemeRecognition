# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:48:21 2019

@author: isaac
"""

import tensorflow as tf
from grelurbm import GReluRBM

""" Get tf variables as numpy vars """
grelurbm = GReluRBM(n_visible=100, n_hidden=25, momentum = 0, l1 = 0.001, use_tqdm=False, err_function = 'rmse')
grelurbm.load_weights("./rbmrelucats/vowel_semivowel/", "weights.h5")
w1 = grelurbm.sess.run(grelurbm.w)
vb1 = grelurbm.sess.run(grelurbm.visible_bias)
hb1 = grelurbm.sess.run(grelurbm.hidden_bias)
#grelurbm.close_session()

grelurbm.load_weights("./rbmrelucats/fricatives_affricatives_nasals/", "weights.h5")
w2 = grelurbm.sess.run(grelurbm.w)
vb2 = grelurbm.sess.run(grelurbm.visible_bias)
hb2 = grelurbm.sess.run(grelurbm.hidden_bias)
grelurbm.close_session()
del(grelurbm)

class MergeWeights():
    
    def __init__(self, n_visible, n_hidden):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.w1 = tf.placeholder(shape = [None,None], name = "w1", dtype = tf.float32)
            self.w2 = tf.placeholder(shape = [None,None], name = "w2", dtype = tf.float32)
            self.input_bias1 = tf.placeholder(shape = [None], name = "input_bias1", dtype = tf.float32)
            self.input_bias2 = tf.placeholder(shape = [None], name = "input_bias2", dtype = tf.float32)
            self.hidden_bias1 = tf.placeholder(shape = [None], name = "hidden_bias1", dtype = tf.float32)
            self.hidden_bias2 = tf.placeholder(shape = [None], name = "hidden_bias2", dtype = tf.float32)
            
#                     tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
            self.visible_bias = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
            self.hidden_bias = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)
            self.w = tf.Variable(tf.zeros([n_visible, n_hidden]), dtype=tf.float32)

            self.concat_ops = [self.w.assign(tf.concat([self.w1, self.w2], 1)),
                                   self.visible_bias.assign(self.input_bias1),
                                   self.hidden_bias.assign(tf.concat([self.hidden_bias1, self.hidden_bias2],0))]
        
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.graph)

    def merge(self,w1,w2,ib1,ib2,hb1,hb2):
        self.sess.run(self.init)
        self.sess.run(self.concat_ops, feed_dict={self.w1: w1, self.w2:w2,
                                                      self.input_bias1: ib1, self.input_bias2: ib2,
                                                      self.hidden_bias1: hb1, self.hidden_bias2:hb2})

    def get_vars(self):
        w = self.sess.run(self.w)
        vb = self.sess.run(self.visible_bias)
        hb = self.sess.run(self.hidden_bias)
        return w,vb,hb
        
    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)
    
mw = MergeWeights(n_visible = 100, n_hidden = 50)
mw.merge(w1,w2,vb1,vb2,hb1,hb2)
variabs = mw.get_vars()

mw.save_weights("./rbmrelucats/","weights.h5")