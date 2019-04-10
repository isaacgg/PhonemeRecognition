# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:33:26 2019

@author: isaac
"""

from grelurbm import GReluRBM
import DataInfoExtractor as die

from scipy import signal
import numpy as np
import pandas as pd
import utils

#phs_cat = ["vowel", "semivowel", "fricatives", "affricatives", "stop", "nasals"] #forget about others
phs_cat = ["vowel_semivowel", "fricatives_affricatives_nasals"] #forget about others

ie = die.DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")
train_phns = ie.get_train_df()
test_phns = ie.get_test_df()

def create_matrix_from_category(df, winsize, winstride, window_type = lambda x:np.ones((x,))):
    def window_phon(wav):
        result = utils.frame_wav(wav, winsize, winstride, window_type)

        if result.size > 0:
            return result/0.1 #-mean/std
        else:
            return None
        
    data_matrix = []
    for ix, row in df.iterrows():
        wav = row["wav"]
        frames = window_phon(wav)
        if frames is not None:
            data_matrix.extend(frames)
    return np.array(data_matrix, dtype = np.float)

if __name__ == "__main__":
    import DataInfoExtractor as die

    winsize = 400
    winstride = 160
    n_hidden = 600
    window = signal.hann#(winsize) #signal.hann(winsize)#np.ones(winsize)
    lr = 0.001
    batchsize = 128
    epochs = 20
    
    if True: #train or not
        ie = die.DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")
        for cat in phs_cat:
            if cat == "vowel_semivowel":
                df = train_phns[(train_phns["category"]=="vowel") | (train_phns["category"]=="semivowel")]
                n_hidden = 400
        
            if cat == "fricatives_affricatives_nasals":
                df = train_phns[(train_phns["category"]=="fricatives") | (train_phns["category"]=="affricatives") | (train_phns["category"]=="nasals")]
                n_hidden = 200
                
            def lr_scheduler(x,e,d, epochs = epochs, lr_init = lr):
                if e != 0 and e != (epochs-1):
                    x *= (1-(e/epochs)**8)/((1-((e-1)/epochs)**8)) 
                else:
                    x *= (1-(e/epochs)**8)
                if d<0:
                    x /= 2
                return x
                    
            losgdir = "./rbmrelucats"+"/hann/" + cat + "/"
            train_matrix = create_matrix_from_category(df, winsize, winstride, window)
#            train_matrix = ie.get_train_matrix_from_wavs(num_vis, winstride)

            grelurbm = GReluRBM(n_visible=winsize, n_hidden=n_hidden, momentum = 0, l1 = 0, use_tqdm=False, err_function = 'rmse')
            errs = grelurbm.fit(train_matrix, n_epoches=epochs, batch_size=batchsize, learning_rate=lr, lr_scheduler = lr_scheduler)
            
            grelurbm.save_weights(losgdir, "weights.h5")
            grelurbm.close_session()
            del(grelurbm)

    #Comporobar si está bien
    raise Exception

    losgdir = "./rbmrelucats"+"/hann/" + phs_cat[1] + "/"

    grelurbm = GReluRBM(n_visible=winsize, n_hidden=200, momentum = 0, l1 = 0, use_tqdm=False, err_function = 'rmse')
    grelurbm.load_weights(losgdir, "weights.h5")
    
    utils.plot_weights(grelurbm, 50, (30,60))
    utils.plot_input_bias(grelurbm)
    utils.plot_hidden_bias(grelurbm)



#    def plot_weights(rbm, rows):
#        cols = int(np.ceil(rbm.n_hidden/rows))
#        fig, axes = plt.subplots(rows, cols, figsize=(12,12), sharex=True, sharey=True)
#        plot_num = 0
#        
#        _ws = rbm.get_weights()[0]
#        for i in range(rows):
#            for j in range(cols):
#                axes[i,j].plot(np.real(_ws[:,plot_num]))
#                plot_num += 1
#        plt.show()
#
#    def plot_input_bias(rbm):
#        _ws = rbm.get_weights()[1]
#        plt.stem(_ws)
#        plt.show()
#    
#    def plot_hidden_bias(rbm):    
#        _ws = rbm.get_weights()[2]
#        plt.stem(_ws)
#        plt.show()
     
#    grelurbm = GReluRBM(n_visible=num_vis, n_hidden=num_hid, momentum = 0, l1 = 0.001, use_tqdm=False, err_function = 'rmse', lr_scheduler = lr_scheduler)
#    grelurbm.load_weights("./rbmrelucats/vowel_semivowel/", "weights.h5")
#    plot_weights(grelurbm, 5)
#    plot_hidden_bias(grelurbm)
#    plot_input_bias(grelurbm)

#plt.plot(errs)
#plt.show()

#plot_weights(grelurbm, 10)
#plot_input_bias(grelurbm)
#plot_hidden_bias(grelurbm)

  