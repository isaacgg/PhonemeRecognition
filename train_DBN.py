# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:02:22 2019

@author: isaac
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from grelurbm import GReluRBM
from relurelurbm import ReluReluRBM
from gbrbm import GBRBM
from relubrbm import ReluBRBM

class DBN():
    def __init__(self, n_visibles, n_hidden, v=1, momentum = 0, l1 = 0, model = "grelu", logsdir = None):
        self.mean = 0
        self.std = 1
        if model == "grelu":
            self.dbn = GReluRBM(n_visible=n_visibles, n_hidden=n_hidden, momentum=momentum, l1= l1, err_function = 'rmse', logsdir = logsdir)
        elif model == "relurelu":
            self.dbn = ReluReluRBM(n_visible=n_visibles, n_hidden=n_hidden, momentum=momentum, l1= l1, err_function = 'rmse', logsdir = logsdir)
        elif model == "relub":
            self.dbn = ReluBRBM(n_visible=n_visibles, n_hidden=n_hidden, momentum=momentum, err_function = 'rmse', sample_visible = True, l1= l1, logsdir = logsdir)
        elif model == "gb":
            self.dbn = GBRBM(n_visible=n_visibles, n_hidden=n_hidden, momentum=momentum, err_function = 'rmse', sample_visible = True, l1= l1, logsdir = logsdir)
        
        if (model == "grelu") or (model == "relurelu"):
            if v==1:
                self.feats_func = self.dbn.get_features
            elif v==2:
                self.feats_func = self.dbn.get_features_v2
            elif v==3:
                self.feats_func = self.dbn.get_features_v3
            elif v == 4:
                self.feats_func = self.dbn.get_features_v4
        else:
            self.feats_func = self.dbn.get_features
            
            
    def train(self, X_train, lr, dropout, n_epochs,logsdir):
        
        def lr_scheduler(x,e,d, epochs = n_epochs, lr_init = lr):
#            if e != 0 and e != (epochs-1):
#                x *= (1-(e/epochs)**8)/((1-((e-1)/epochs)**8)) 
#            else:
#                (1-(e/epochs)**8)
#            if d<(0 - 1e-5):
#                x *= (3/4)
            return x
        
        self.dbn.fit(np.array(X_train), n_epoches=n_epochs, batch_size=128, learning_rate=lr, lr_scheduler = lr_scheduler, dropout = dropout, correct_hidden = False, correct_visible = False)
        self.dbn.save_weights(logsdir, "weights.h5")

    def set_metrics(self, mean = 0, std = 1):
        self.mean = mean
        self.std = std

    def load_weights(self, logsdir):
        self.dbn.load_weights(logsdir, "weights.h5")
    
    def close_session(self):
        self.dbn.close_session()
        del(self.dbn)
        
    def extract_feat(self,x):
        frames = [(f-self.mean)/self.std for f in x]
        return self.feats_func(frames)

    def plot(self):
        def plot_weights(rows):
            cols = int(np.ceil(self.dbn.n_hidden/rows))       #12*12
            fig, axes = plt.subplots(rows, cols, figsize=(60,60), sharex=True, sharey=True)
            plot_num = 0
        
            _ws = self.dbn.get_weights()[0]
            for i in range(rows):
                for j in range(cols):
                    axes[i,j].plot(np.real(_ws[:,plot_num]))
                    plot_num += 1
            plt.show()
        
        def plot_input_bias():
            _ws = self.dbn.get_weights()[1]
            plt.stem(_ws)
            plt.show()
        
        def plot_hidden_bias():    
            _ws = self.dbn.get_weights()[2]
            plt.stem(_ws)
            plt.show()
            
def normalize_fn(train_matrix, logdir_feats):
    try:
        mean = utils.open_pickle(logdir_feats + "mean.pkl")
    except:
        mean = np.array(train_matrix).mean(0)
        utils.save_pickle(logdir_feats + "mean.pkl", mean)
    try:
        std = utils.open_pickle(logdir_feats + "std.pkl")
    except:    
        std = np.array(train_matrix).std(0)
        utils.save_pickle(logdir_feats + "std.pkl", std)
        
    train_matrix = np.array([(r-mean)/(std) for r in train_matrix])
    return train_matrix

#def train():
#    
#def finetune():
#    
#def 3dplot():
#    
#def extractfeats():


if __name__ == "__main__":
    import os    
    
    import Floyd.scripts.commons.utils as utils
    """ CONF """
#    STEP 1
    n_visibles = 100
    n_hidden = 70
    model = "grelu" #relurelu, grelu, gb, relub
    
#    root_dir = "./PreprocessedData/RBMs/"
    root_dir = "./PreprocessedData/RBMs_longtrain/"

    win_dirs = ["hann", "boxcar"]
    w = "hann"
    v=1
    momentum = 0
    l1 = 0.000
    dropout = 1
    normalize = True
    logdir_feats = root_dir + w + "/v2/400_"+str(n_visibles)+"/"


#    STEP 2
#    n_visibles = 500
#    n_hidden = 420
#    model = "grelu" #relurelu, grelu, gb, relub
#
#    
#    root_dir = "./PreprocessedData/RBMs/"
#    win_dirs = ["hann", "boxcar"]
#    w = "hann"
#    v = 2
#    momentum = 0
#    l1 = 0.000
#    dropout = 1
#    normalize = True
#    logdir_feats = root_dir + w + "/v2/400_600/relurelu_600_500/relurelu_v2/"
    
##    STEP 3
#    n_visibles = 200
#    n_hidden = 100
#    model = "relurelu"
#    
#    root_dir = "./PreprocessedData/RBMs/"
#    win_dirs = ["hann", "boxcar"]
#    w = "hann"
#    v = 1
#    momentum = 0.5
#    l1 = 0.00
#    dropout = 0.05
#    logdir_feats = root_dir + w + "/v2/400_600/v1/600_300/300_200/relurelu_v1/"
    
    #NOT WORKING
##    STEP 4
#    n_visibles = 100
#    n_hidden = 75
#    model = "relurelu"
#    
#    root_dir = "./PreprocessedData/RBMs/"
#    win_dirs = ["hann", "boxcar"]
#    w = "hann"
#    v = 1
#    momentum = 0.5
#    l1 = 0.00
#    dropout = 0.05
#    logdir_feats = root_dir + w + "/v2/400_600/v1/600_300/300_200/relurelu_v1/200_100/relurelu_v1/"

    i = 4
    weights_dir = logdir_feats + model + "_" + str(n_visibles) + "_" + str(n_hidden) + "/"
    out_feats_dir = weights_dir + model + "_v" + str(v) + "/" 

    lr = 0.001 #0.0005087695972171545
##    """ TRAIN """
    if i == 1:
        from utils import plot_weights, plot_input_bias, plot_hidden_bias
        n_epochs = int(200//dropout)
        print("train for: ", str(n_epochs))

        X_train = utils.load_npy(logdir_feats + "X_train.npy")
        train_matrix = []
        for x in X_train:
            train_matrix.extend(x)
        train_matrix = np.array(train_matrix)
        del(X_train)

        if normalize:
            train_matrix = normalize_fn(train_matrix, logdir_feats)
#            try:
#                mean = utils.open_pickle(logdir_feats + "mean.pkl")
#            except:
#                mean = np.array(train_matrix).mean(0)
#                utils.save_pickle(logdir_feats + "mean.pkl", mean)
#            try:
#                std = utils.open_pickle(logdir_feats + "std.pkl")
#            except:    
#                std = np.array(train_matrix).std(0)
#                utils.save_pickle(logdir_feats + "std.pkl", std)
#                
#            train_matrix = np.array([(r-mean)/(std) for r in train_matrix])

        dbn = DBN(n_visibles,n_hidden, v, momentum, l1=l1, model = model, logsdir = weights_dir)
        dbn.train(train_matrix, lr, dropout = dropout, n_epochs = n_epochs, logsdir = weights_dir)
        del(train_matrix)

        plot_weights(dbn.dbn, 20)
        plot_input_bias(dbn.dbn)
        plot_hidden_bias(dbn.dbn)

        dbn.close_session()
        del(dbn)
#dbn.dbn.save_weights(weights_dir, "weights.h5")

    elif i == 2:
        from utils import plot_weights, plot_input_bias, plot_hidden_bias
        n_epochs = 500
        print("train for: ", str(n_epochs))
        dropout = 1

        X_train = utils.load_npy(logdir_feats + "X_train.npy")
        train_matrix = []
        for x in X_train:
            train_matrix.extend(x)
        train_matrix = np.array(train_matrix)
        del(X_train)

        if normalize:
            train_matrix = normalize_fn(train_matrix, logdir_feats)

        dbn = DBN(n_visibles,n_hidden, v, momentum, l1=l1, model = model, logsdir = weights_dir)
        dbn.load_weights(weights_dir)
#        dbn.dbn.fix_hidden_bias()
        dbn.train(train_matrix, lr, dropout = dropout, n_epochs = n_epochs, logsdir = weights_dir)
        del(train_matrix)

        plot_weights(dbn.dbn, 20)
        plot_input_bias(dbn.dbn)
        plot_hidden_bias(dbn.dbn)

        dbn.close_session()
        del(dbn)
#dbn.dbn.save_weights(weights_dir, "weights.h5")
 
#    """ PLOT """
    elif i ==3:
        """ Crea un scatterplot 3D """
#        from utils import collapse_num_labels, plot_3d, labels_to_str, labels_num_to_category
        X_train = utils.open_pickle(root_dir + w + "/v1/400_600/"  + "X_train.npy")
        X_train = X_train[:500]
        feats = []
        for x in X_train:
            feats.extend(x)
        del(X_train)

#        X_train = load_pkl(logdir_feats + "X_train.npy")
#        X_train = X_train[:500]
#
#        dbn = DBN(n_visibles,n_hidden, v, momentum, l1=l1, model = model)
#        dbn.load_weights(weights_dir)
#
#        if normalize:
#            with open(logdir_feats + "mean.pkl", 'rb') as handle:
#                mean = pickle.load(handle)
#            with open(logdir_feats + "std.pkl", 'rb') as handle:
#                std = pickle.load(handle)
#            dbn.set_metrics(mean, std)
#
#        feats = []
#        for x in X_train:
#            feats.extend(dbn.extract_feat(x))
#        del(X_train)
#        dbn.close_session()
#        del(dbn)

        y_train = utils.open_pickle(root_dir + w + "/v1/400_600/" + "y_train.npy")
        y_train = y_train[:500]
        labels = []
        for y in y_train:
            labels.extend(utils.collapse_num_labels(y))
        del(y_train)

        from PrincipalComponentAnalysis import PCA
        pca = PCA().compute_pca(feats,3)
    
        utils.plot_3d(pca,utils.labels_to_str(labels))
#        plot_3d(pca,labels_num_to_category(labels))
        
        del(labels)
        del(feats)
        del(pca)
        
    if i == 4:        
        """ Crea una matriz de scatterplots """
        import pandas as pd
        import utils
 
        category = "category" #category, phon
        labels_to_plot = []#"fricative", "affricative"]#["aa", "ah"]
       
        X_train = utils.load_npy(root_dir + w + "/v1/400_600/"  + "X_train.npy")
        X_train = X_train[:500]
        feats = []
        for x in X_train:
            feats.extend(x)
        del(X_train)
        
#        X_train = load_pkl(logdir_feats + "X_train.npy")
#        X_train = X_train[:500]
#        
#        dbn = DBN(n_visibles,n_hidden, v, momentum, l1=l1, model = model)
#        dbn.load_weights(weights_dir)
#        
#        if normalize:
#            with open(logdir_feats + "mean.pkl", 'rb') as handle:
#                mean = pickle.load(handle)
#            with open(logdir_feats + "std.pkl", 'rb') as handle:
#                std = pickle.load(handle)
#            dbn.set_metrics(mean, std)
#        
#        feats = []
#        for x in X_train:
#            feats.extend(dbn.extract_feat(x))
#        del(X_train)
#        dbn.close_session()
#        del(dbn)
        
        
        y_train = utils.load_npy(root_dir + w + "/v2/400_50/" + "y_train.npy")
        y_train = y_train[:500]
        labels = []
        for y in y_train:
            labels.extend(utils.collapse_num_labels(y))
        del(y_train)
        
        phons_df = pd.DataFrame(columns = ["phon", "feats", "category"])
        phons_df['phon'] = utils.labels_to_str(labels, collapsed = True)
        phons_df['feats'] = feats
        phons_df['category'] = utils.labels_num_to_category(labels)
      
        if labels_to_plot != []:
            phons_df = phons_df[phons_df[category].isin(labels_to_plot)]

        labels_to_plot = phons_df[category].unique()            
        
        from PrincipalComponentAnalysis import PCA
        pca = PCA().compute_pca(list(phons_df['feats'].values), 2)
        
        #TODO: Crear un df pca
        df_to_plot = pd.DataFrame(pca, columns = ["pca0", "pca1"])
        df_to_plot[category] = phons_df[category].values
                
        utils.plot_scatter_matrix(df_to_plot, category, labels_to_plot)
        
        del(feats)
        del(labels_to_plot)
        del(phons_df)
        del(pca)
#        utils.plot_scatter_matrix(phons_df, "phon", ["ah","b","sil"])

#    """ EXTRACT FEATURES """    
    if i == 5:
        try:
            os.makedirs(out_feats_dir)
        except FileExistsError:
            print("ya existe el directorio!!")
            print(out_feats_dir)
        
        X_file = "X_test.npy"
        X = utils.open_pickle(logdir_feats + X_file)

        dbn = DBN(n_visibles,n_hidden, v, momentum, l1=l1, model = model)
        dbn.load_weights(weights_dir)
        
        if normalize:
            with open(logdir_feats + "mean.pkl", 'rb') as handle:
                mean = pickle.load(handle)
            with open(logdir_feats + "std.pkl", 'rb') as handle:
                std = pickle.load(handle)        
            dbn.set_metrics(mean, std)

        feats = []
        for x in X:
            feats.append(dbn.extract_feat(x))
        del(X)
        dbn.close_session()
        del(dbn)
        
        utils.save_pickle(out_feats_dir + X_file, np.array(feats))
        del(feats)