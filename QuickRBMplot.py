# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:05:28 2019

@author: isaac
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
from scipy import signal


import python_speech_features as psf
from grelurbm import GReluRBM

from utils import read_wav as wavread
from utils import plot_3d

window=signal.hann
winstep=160
winlen=400
n_hidden=600
weightsdir='./prueba/' 
logsdir = "./dataAnalisis/greluRBM_400_600/"

phns_all = ["h#","sh","ix","hv","eh","dcl","jh","ih","d","ah","kcl","k","s","ux","q","en","gcl","g","r","w","ao","epi"
             ,"dx","axr","l","y","uh","n","ae","m","oy","ax","dh","tcl","iy","v","f","t","pcl","ow","hh","ch","bcl","b",
             "aa","em","ng","ay","th","ax-h","ey","p","aw","er","nx","z","el","uw","pau","zh","eng"]


phns_cat_dict = {"h#":"other","sh":"fricatives","ix":"vowel","hv":"semivowel","eh":"vowel","dcl":"stop",
                 "jh":"affricative", "ih":"vowel","d":"stop","ah":"vowel","kcl":"stop","k":"stop","s":"fricatives",
                 "ux":"vowel","q":"stop","en":"nasals","gcl":"stop","g":"stop","r":"semivowel", "w":"semivowel",
                 "ao":"vowel","epi":"other","dx":"stop","axr":"vowel","l":"semivowel","y":"semivowel","uh":"vowel",
                 "n":"nasals","ae":"vowel","m":"nasals","oy":"vowel","ax":"vowel","dh":"fricatives", "tcl":"stop",
                 "iy":"vowel","v":"fricatives","f":"fricatives","t":"stop","pcl":"stop","ow":"vowel","hh":"semivowel",
                 "ch":"affricative","bcl":"stop", "b":"stop", "aa":"vowel","em":"nasals","ng":"nasals","ay":"vowel",
                 "th":"fricatives","ax-h":"vowel","ey":"vowel","p":"stop","aw":"vowel","er":"vowel","nx":"nasals",
                 "z":"fricatives","el":"semivowel","uw":"vowel","pau":"other","zh":"fricatives","eng":"nasals"}

collapse_dict = {"ao":"aa", "ax":"ah", "ax-h":"ah", "axr": "er", "hv":"hh", "ix":"ih", "el":"l", "em":"m", "en":"n",
                 "nx":"n", "eng":"ng","zh":"sh", "ux":"uw","pcl":"sil", "tcl":"sil", "kcl":"sil", "bcl":"sil", 
                 "dcl":"sil", "gcl":"sil", "h#":"sil", "pau":"sil", "epi":"sil", "q":"-"}

root_wavs_dir = "./Database/TIMIT/TRAIN"
checkpoints_folder = "./data_checkpoints"

phn_files = glob.glob(root_wavs_dir + "/*/*/*.PHN")

def read_wav(wav_file):
    fr, wav = wavread(wav_file)
    wav = wav/np.max(np.abs(wav))
    return wav, fr

def read_phons(phn_file):
    phn_list = []
    for line in open(phn_file):
        elems = line.split(" ")
        phn_list.append({"start": int(elems[0]), "end": int(elems[1]), "phn": elems[2].rstrip()})
    return phn_list

def extract_phons(wav, phns_list):
    phns_df = pd.DataFrame(columns = ["wav", "phn", "phn_index", "category", "collapsed"])
    for phn in phns_list:
        phn_wav = wav[int(phn["start"]):int(phn["end"]+1)]
        phn_index = phns_all.index(phn["phn"])
        phn_label = phn["phn"]
        category = phns_cat_dict[phn_label]
        try:
            collapsed = collapse_dict[phn_label]
        except:
            collapsed = phn_label
        phns_df = phns_df.append({"wav": phn_wav, "phn": phn_label, "phn_index": phn_index,
                                  "category": category,"collapsed":collapsed}, ignore_index = True)
    return phns_df

def create_dict(phns_files):
    _phns_df = pd.DataFrame(columns = ["wav", "phn", "phn_index", "category", "collapsed"])
    for phn_file in phns_files:
        wav_file = phn_file[:-4] + "RIFF.wav"
        wav,_ = read_wav(wav_file)
        phns_list = read_phons(phn_file)
        _phns_df = _phns_df.append(extract_phons(wav, phns_list), ignore_index = True)
    return _phns_df

#Read phonemes DataFrame, create if it doesn't exists
phns_df_file = os.path.join(checkpoints_folder, "phns_df.pkl")
if os.path.isfile(phns_df_file):
    phns_df = pd.read_pickle(phns_df_file)
else:    
    phns_df = create_dict(phn_files)
    pd.to_pickle(phns_df, path = phns_df_file)

#Select only a few samples to speed up the process
def create_samples_df(df, n = 500):
    samples_df = pd.DataFrame(columns = df.columns)
    for p in df["collapsed"].unique():
        to_append = df[df["collapsed"]== p].sample(n=n)
        samples_df = samples_df.append(to_append, ignore_index=True)
    return samples_df

samples_df = create_samples_df(phns_df, 500)

with open("D:/TrabajoFinDeMaster/greluRBM_400_600_hann/" + "mean.pkl", 'rb') as handle:
    mean = pickle.load(handle)
with open("D:/TrabajoFinDeMaster/greluRBM_400_600_hann/" + "std.pkl", 'rb') as handle:
    std = pickle.load(handle)

    
grelurbm = GReluRBM(n_visible=winlen, n_hidden=n_hidden, momentum=0, use_tqdm=False, err_function = 'rmse')
grelurbm.load_weights(weightsdir, "weights.h5")


feats_df = pd.DataFrame(columns=["phn", "feats", "algth", "idx"])
for ix,row in samples_df.iterrows():
    label = row["phn"]
    frames = psf.sigproc.framesig(sig=row["wav"], frame_len=winlen, frame_step=winstep)
    frames = [window(winlen)*(f-mean)/std for f in frames]
    
    if frames == []:
        continue
    f_rbm = grelurbm.get_features_v2(frames) #v2 es relu sin bias, v3 es mf con bias
    feats_df = feats_df.append({"phn": label, "category":row["category"], "collapsed":row["collapsed"],
                "feats": f_rbm, "algth": "rbm_relu", "idx": ix}, ignore_index = True)

grelurbm.close_session()
del(grelurbm)

from PrincipalComponentAnalysis import PCA
feats = []
for ix,row in feats_df.iterrows():
    feats.extend(row["feats"])
    
pca = PCA().compute_pca(feats,3)

plot_3d(pca, feats_df["collapsed"].values)
