# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 16:26:17 2018

@author: isaac
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(wav, title = ""):
    t = np.arange(0, len(wav))
    plt.plot(t, wav)

    plt.xlabel('Frames')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plotscatter(feats, title = ''):
    fig, ax = plt.subplots()
    feats= np.swapaxes(feats, 0 ,1)
    cax = ax.imshow(feats, interpolation='nearest', cmap=cm.seismic, origin='lower')
    ax.set_title(title)
    plt.xlabel("n_windows")
    plt.show()
    
def plotbar(amounts, labels, title = "", xlabel = "labels", ylabel = "amounts"):
    # this is for plotting purpose
    index = np.arange(len(labels))
    plt.bar(index, amounts)
    plt.xlabel(xlabel, fontsize=5)
    plt.ylabel(ylabel, fontsize=5)
    plt.xticks(index, labels, fontsize=5, rotation=30)
    plt.title(title)
    plt.show()