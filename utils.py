# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:59:43 2019

@author: isaac
"""

from itertools import groupby
import pickle
import numpy as np
import python_speech_features as psf
from scipy.io.wavfile import read as wavread
from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns


""" Signal """
def read_wav(wav_file):
    fr, wav = wavread(wav_file)
    wav = wav/np.max(np.abs(wav))
    return wav, fr

def frame_wav(wav, winlen, winstep, winfunc = lambda x:np.ones((x,))):
    return psf.sigproc.framesig(sig=wav, frame_len=winlen, frame_step=winstep, winfunc = winfunc)

"""Data"""
def load_npy(file):
    return np.load(file)

def save_npy(data, file):
    np.save(arr = data, file = file)

def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(file, data):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)  

def get_mini_dataset(X, y, n):
    n_data = len(X)
    inds = np.arange(n_data)
    np.random.shuffle(inds)
    X_cpy = X[inds]
    y_cpy = y[inds]
    return X_cpy[:n], y_cpy[:n]

""" Labels and TIMIT related"""
from Floyd.scripts.commons.timit_utils import *
#phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", "f",
#            "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih", "eh", "ey",
#            "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]
#
#collapse_keys = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'h', 'ih', 'iy', 'jh', 'k', 'l', 'm',
#                 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z','-']
#
#collapse_dict = {"aa":"aa", "ao":"aa", "ah":"ah", "ax":"ah", "ax-h":"ah", "er":"er", "axr": "er", "hh":"h", "hv":"h", "jh":"jh", "d":"d",
#                 "k":"k", "s":"s", "g":"g", "r":"r", "w":"w", "dx":"dx", "y":"y", "uh":"uh", "ae":"ae", "oy":"oy", "dh":"dh", "iy":"iy",
#                 "v":"v", "f":"f", "t":"t", "ow":"ow", "ch":"ch", "b":"b", "ay":"ay", "th":"th", "ey":"ey", "p":"p", "aw":"aw", "z":"z",
#                 "eh":"eh", "ih":"ih","ix":"ih", "el":"l", "l":"l", "em":"m", "m":"m", "en":"n", "n":"n", "nx":"n", "eng":"ng","ng":"ng",
#                 "zh":"sh", "sh":"sh", "ux":"uw", "uw":"uw", "pcl":"sil", "tcl":"sil", "kcl":"sil", "bcl":"sil", "dcl":"sil", "gcl":"sil",
#                 "h#":"sil", "pau":"sil", "epi":"sil", "q":"-"}
#                 
#
#
##phns_cat_dict = {"h#":"other","sh":"fricative","ix":"vowel","hv":"semivowel","eh":"vowel","dcl":"stop",
##                 "jh":"affricative", "ih":"vowel","d":"stop","ah":"vowel","kcl":"stop","k":"stop","s":"fricative",
##                 "ux":"vowel","q":"stop","en":"nasal","gcl":"stop","g":"stop","r":"semivowel", "w":"semivowel",
##                 "ao":"vowel","epi":"other","dx":"stop","axr":"vowel","l":"semivowel","y":"semivowel","uh":"vowel",
##                 "n":"nasal","ae":"vowel","m":"nasal","oy":"vowel","ax":"vowel","dh":"fricative", "tcl":"stop",
##                 "iy":"vowel","v":"fricative","f":"fricative","t":"stop","pcl":"stop","ow":"vowel","h":"semivowel",
##                 "ch":"affricative","bcl":"stop", "b":"stop", "aa":"vowel","em":"nasal","ng":"nasal","ay":"vowel",
##                 "th":"fricative","ax-h":"vowel","ey":"vowel","p":"stop","aw":"vowel","er":"vowel","nx":"nasal",
##                 "z":"fricative","el":"semivowel","uw":"vowel","pau":"other","zh":"fricative","eng":"nasal"}
#
#phns_39_cat_dict = {'t': 'plosives', 'g': 'plosives', 'k': 'plosives','jh': 'plosives',
#                    'b': 'plosives', 'd': 'plosives', 'p': 'plosives', 'ch': 'plosives',
#                    'h': 'fricatives', 'sh': 'fricatives', 'z': 'fricatives', 'f': 'fricatives',
#                    'v': 'fricatives', 'dh': 'fricatives', 's': 'fricatives', 'th': 'fricatives',
#                    'm': 'nasals', 'n': 'nasals', 'ng': 'nasals', 
#                    'r': 'semivowels', 'w': 'semivowels', 'er': 'semivowels', 'l': 'semivowels', 'y': 'semivowels',
#                    'uh': 'vowels', 'ae': 'vowels', 'ih': 'vowels', 'iy': 'vowels',
#                    'eh': 'vowels', 'ah': 'vowels', 'aa': 'vowels', 'uw': 'vowels',
#                    'aw': 'diphthongs', 'ay': 'diphthongs', 'oy': 'diphthongs', 'ow': 'diphthongs', 'ey': 'diphthongs',
#                    'dx': 'closures', 'sil': 'closures'}
#
#categories_key = ['plosives', 'fricatives', 'nasals', 'semivowels', 'vowels', 'diphthongs', 'closures']
#
#
#
##cats = {'vowels': {'aw', 'er', 'ow', 'l', 'aa', 'ae', 'ah', 'y', 'uw', 'ey', 'ay', 'uh', 'r', 'ih', 'iy', 'w', 'oy', 'eh'},
##         'stops': {'t', 'g', 'k', 'jh', 'b', 'd', 'p', 'ch'},
##         'fricatives': {'sh', 'z', 'f', 'v', 'h', 'dh', 's', 'th'},
##         'nasals': {'n', 'ng', 'm'},
##         'silences': {'sil', 'dx', '-'}}
#
#phns_39_cat_dict = {'aw': 'vowels', 'ay': 'vowels', 'uh': 'vowels', 'r': 'vowels', 'w': 'vowels', 'oy': 'vowels', 
#                    'er': 'vowels', 'ae': 'vowels', 'ih': 'vowels', 'iy': 'vowels', 'l': 'vowels', 'ow': 'vowels',
#                    'ah': 'vowels', 'eh': 'vowels', 'y': 'vowels', 'aa': 'vowels', 'uw': 'vowels', 'ey': 'vowels',
#                    't': 'stops', 'g': 'stops', 'k': 'stops','jh': 'stops',
#                    'b': 'stops', 'd': 'stops', 'p': 'stops', 'ch': 'stops',
#                    'h': 'fricatives', 'sh': 'fricatives', 'z': 'fricatives', 'f': 'fricatives',
#                    'v': 'fricatives', 'dh': 'fricatives', 's': 'fricatives', 'th': 'fricatives',
#                    'm': 'nasals', 'n': 'nasals', 'ng': 'nasals',
#                    'dx': 'silences', 'sil': 'silences', '-': 'silences'}
#
#cats_keys = ['vowels', 'stops', 'fricatives', 'nasals', 'silences']
#
#new_dict = {}
#for c in cats.keys():
#    for f in cats[c]:
#        new_dict[f] = c
#
##cats = {"vowels": {"aa", "ae", "ah", "ao", "ax", "ax-h", "axr", "ay", "aw", "eh", "el", "er",
##               "ey", "ih", "ix", "iy", "l", "ow", "oy", "r", "uh", "uw", "ux", "w", "y"},
##"stops":{"p", "t", "k", "b", "d", "g", "jh", "ch"},
##"fricatives": {"s", "sh", "z", "zh", "f", "th", "v", "dh", "hh", "hv"},
##"nasals": {"m", "em", "n", "nx", "ng", "eng", "en"},
##"silences": {"h#", "epi", "pau", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "q", "dx"}}
#
#
#             
#new_dict = {}
#for c in cats.keys():
#    phons_list = []
#    for f in cats[c]:
#        phons_list.append(collapse_dict[f])
#    phons_list = list(dict.fromkeys(phons_list))
#    new_dict[c] = phons_list
# 
##phns_39_dict = {"aa": ["aa", "ao"],
##                "ah": ["ah", "ax", "ax-h"],
##                "er": ["er", "axr"],
##                "hh": ["hh", "hv"],
##                "ih": ["ih", "ix"],
##                "l": ["l", "el"],
##                "m": ["m", "em"],
##                "n": ["n", "en", "nx"],
##                "ng": ["ng", "eng"],
##                "sh": ["sh", "zh"],
##                "uw": ["uw", "ux"],
##                "sil": ["pcl", "tcl", "kcl", "bcl", "dcl", "gcl", "h#", "pau", "epi"],
##                "-": ["q"]}
#
#def remove_q(y):
#    y = np.array(y)
#    new_y = []
#    index_to_remove = collapse_keys.index("-")
#    for row in y:
#        row = [x for x in row if x != index_to_remove]
#        new_y.append(row)
#    return np.array(new_y)
#
#def to_ctc(y):
#    y = np.array(y)
#    new_y = []
#    for row in y:
#        new_y.append([k for k,g in groupby(row)])
#    
#    return np.array(new_y)
#
#def labels_to_int(labels, collapsed = True):
#    if collapsed:
#        return [collapse_keys.index(y) for y in labels]
#    else:
#        return [phonemes.index(y) for y in labels]
#
#def labels_to_str(labels, collapsed = True):
#    if collapsed:
#        return [collapse_keys[y] for y in labels]
#    else:
#        return [phonemes[y] for y in labels]
#    
#def collapse_str_labels(labels):
#    return [collapse_dict[y] for y in labels]
#    
#def collapse_num_labels(labels):
#    return [collapse_keys.index(collapse_dict[phonemes[int(y)]]) for y in labels]
#
#def labels_str_to_category(labels): #Just for collapsed phons
#    return [phns_39_cat_dict[y] for y in labels]
#
#def labels_num_to_category(labels):
#    return [phns_39_cat_dict[y] for y in labels_to_str(labels)]
#    
#def categories_to_int(categories):
#    return [categories_key.index(c) for c in categories]

    
""" Visualization """
def plot_scatter_matrix(df, category = "phon", labels = [], figsize =(16,16)):
    if labels != []:
        df = df[df[category].isin(labels)]

    labels = df[category].unique()
    
    
    fig,axes = plt.subplots(len(labels), len(labels), figsize = figsize, sharex=True, sharey=True)
    
    for ax, col in zip(axes[0], labels):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], labels):
        ax.set_ylabel(row, rotation=0, size='large')
    
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            df[category]==labels[col]
            axes[row,col].scatter(df[df[category]==labels[col]]["pca0"].values, df[df[category]==labels[col]]["pca1"].values, s = 0.5)
            axes[row,col].scatter(df[df[category]==labels[row]]["pca0"].values, df[df[category]==labels[row]]["pca1"].values, s = 0.5)
#            axes[row,col].set_xlim(0,10)
#            axes[row,col].set_ylim(0,10)
    plt.show()


def plot_weights(rbm, rows, size = (60,60)):
    cols = int(np.ceil(rbm.n_hidden/rows))       #12*12
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)
    plot_num = 0

    _ws = rbm.get_weights()[0]
    for i in range(rows):
        for j in range(cols):
            axes[i,j].plot(np.real(_ws[:,plot_num]))
            plot_num += 1
    plt.show()

def plot_input_bias(rbm):
    _ws = rbm.get_weights()[1]
    plt.stem(_ws)
    plt.show()

def plot_hidden_bias(rbm):    
    _ws = rbm.get_weights()[2]
    plt.stem(_ws)
    plt.show()

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def plot_3d(pca, labels):
    def get_spaced_colors(n):
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        
        return ['rgb(' + str(int(i[:2], 16))+ ',' + str(int(i[2:4], 16)) + ',' + str(int(i[4:], 16)) + ')' for i in colors]

    labels_unique = np.unique(labels)
    if len(labels_unique) <= 10:
        colorslist = DEFAULT_PLOTLY_COLORS
#    elif len(labels_unique) == 40:
#        colorslist = COLLAPSED_COLORS
    else:
        N = len(labels_unique)
#        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
#        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        
        colorslist = get_spaced_colors(N)
#        colorslist = ['rgb(' + str(int(255*c[0]))+ ',' + str(int(255*c[1]))+ ',' 
#                        + str( int(255*c[2])) + ')' for c in RGB_tuples]
        
#        colorslist = DEFAULT_PLOTLY_COLORS + ['rgb(' + str(np.random.randint(0, high=256)) + ',' 
#                      + str(np.random.randint(0, high=256)) + ',' 
#                      + str(np.random.randint(0, high=256)) + ')' for i in range(len(labels_unique)-10)]
        
    data = []
    for ix, p in enumerate(labels_unique):
        ixs = np.where(np.array(labels) == p)[0] #all frames with label p
        trace1 = go.Scatter3d(
            x=pca[ixs,0],
            y=pca[ixs,1],
            z=pca[ixs,2],
            mode='markers',
            marker=dict(
                size=1,
                line=dict(
                    color= colorslist[ix%len(colorslist)],
                    width=10
                ),
                opacity=1
            ),
            name = p
        )
        data.append(trace1)
        
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig)
    
""" Other """
def suspender():
    from ctypes import windll
    # Suspender.
    if not windll.powrprof.SetSuspendState(False, False, False):
        print("No se ha podido suspender el sistema.")
        