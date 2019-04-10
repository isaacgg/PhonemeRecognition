# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:38:34 2018

@author: isaac
"""

import pandas as pd
import pickle
import glob
import os
import numpy as np
from scipy.io.wavfile import read as wavread

import utils as u


"""  """ 
def read_wav(wav_file):
    fr, wav = wavread(wav_file)
    wav = wav/np.max(np.abs(wav))
    return wav, fr   

class DatasetInfoExtractor():
    #All phons
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

    collapsed_all = ['sil', 'sh', 'ih', 'hh', 'eh', 'jh', 'd', 'ah', 'k', 's', 'uw', '-', 'n', 'g', 'r', 'w', 'aa', 'dx',
                     'er', 'l', 'y', 'uh', 'ae', 'm', 'oy', 'dh', 'iy', 'v', 'f', 't', 'ow', 'ch', 'b', 'ng', 'ay', 'th',
                     'ey', 'p', 'aw', 'z']
    
    def __init__(self, dataset_folder, checkpoints_folder):
        self.dataset_folder = dataset_folder
        self.checkpoints_folder = checkpoints_folder
    
    def get_ctc_train_data(self, train_path = 'data_checkpoints/train_dataset.pkl'):
        pkl_file = os.path.join(self.checkpoints_folder, "train_ctc_data_df.pkl")

        if os.path.isfile(pkl_file):
            return pd.read_pickle(pkl_file)
        else:

            train_df = pd.read_pickle(train_path)
        
            train_data_df = pd.DataFrame(columns = ["wav", "phons", "collapsed"])
            
            for ix,row in train_df.iterrows():
                wav,fr = read_wav("./" + row["path"][1:])
                phons_list = row["phons"] 
                
                sorted(phons_list, key = lambda x: x['start'])
                phons = [self.phns_all.index(x['phon']) for x in phons_list]
                collapsed = [self.collapsed_all.index(x['phon']) if x['phon'] not in self.collapse_dict.keys() else self.collapsed_all.index(self.collapse_dict[x['phon']]) for x in phons_list]
                        
                train_data_df = train_data_df.append({"wav":wav, "phons": phons, "collapsed": collapsed}, ignore_index = True)
                
            pd.to_pickle(train_data_df, path = pkl_file)
            return train_data_df
  
    def get_ctc_test_data(self, test_path = 'data_checkpoints/test_dataset.pkl'):
        pkl_file = os.path.join(self.checkpoints_folder, "test_ctc_data_df.pkl")

        if os.path.isfile(pkl_file):
            return pd.read_pickle(pkl_file)
        else:
    
            test_df = pd.read_pickle(test_path)
            
            test_data_df = pd.DataFrame(columns = ["wav", "phons", "collapsed"])
            
            for ix,row in test_df.iterrows():
                wav,fr = read_wav("./" + row["path"][1:])
                phons_list = row["phons"] 
                
                sorted(phons_list, key = lambda x: x['start'])
                phons = [self.phns_all.index(x['phon']) for x in phons_list]
                collapsed = [self.collapsed_all.index(x['phon']) if x['phon'] not in self.collapse_dict.keys() else self.collapsed_all.index(self.collapse_dict[x['phon']]) for x in phons_list]
                
                test_data_df = test_data_df.append({"wav":wav, "phons": phons, "collapsed": collapsed}, ignore_index = True)
    
            pd.to_pickle(test_data_df, path = pkl_file)
            return test_data_df
    
    def get_train_df(self):
        pkl_file = os.path.join(self.checkpoints_folder, "train_phns_df.pkl")
        
        if os.path.isfile(pkl_file):
            return pd.read_pickle(pkl_file)
        else:
            phn_files = glob.glob(os.path.join(self.dataset_folder, "TRAIN/*/*/*.PHN"))
            phns_df = self._create_dict(phn_files)
            pd.to_pickle(phns_df, path = pkl_file)
            return phns_df
                        
    def get_test_df(self):
        pkl_file = os.path.join(self.checkpoints_folder, "test_phns_df.pkl")
        
        if os.path.isfile(pkl_file):
            return pd.read_pickle(pkl_file)
        else:
            phn_files = glob.glob(os.path.join(self.dataset_folder, "TEST/*/*/*.PHN"))
            phns_df = self._create_dict(phn_files)
            pd.to_pickle(phns_df, path = pkl_file)
            return phns_df
    
    def get_n_window_per_phon_train(self, winsize, n):
        train_matrix = []
        
        train_df = self.get_train_df()
        for ix, row in train_df.iterrows():
            wav = row["wav"]
            wavlen = len(wav)
            
            for i in range(n):
                if wavlen >= winsize:
                    start = np.random.randint(0, wavlen-winsize)
                    train_matrix.append(wav[start:start+winsize])
        return train_matrix

    def get_n_window_per_phon_test(self, winsize, n):
        test_matrix = []
        
        test_df = self.get_test_df()
        for ix, row in test_df.iterrows():
            wav = row["wav"]
            wavlen = len(wav)
            
            for i in range(n):
                if wavlen >= winsize:
                    start = np.random.randint(0, wavlen-winsize)
                    test_matrix.append(wav[start:start+winsize])
        return test_matrix

    def get_train_n_random_phons(self, n, winsize, winstride, window_type = lambda x:np.ones((x,)), exclude_silents = True):
        def window_phon(wav):
            result = u.frame_wav(wav, winsize, winstride, window_type)
            if result.size > 0:
                return result
            else:
                return None
                
        train_matrix = []
        train_df = self.get_train_df()
        if exclude_silents:
            samples = train_df[train_df["phn"] != "h#"].sample(n=n)
        else:
            samples = train_df.sample(n=n)

        for idx, s in samples.iterrows():
            result = window_phon(s["wav"])
            while result is None:
                s = train_df.sample(n=1).iloc[0]
                result = window_phon(s["wav"])
                
            train_matrix.extend(result)
        return train_matrix
        
    def get_test_n_random_phons(self, n, winsize, winstride, window_type = lambda x:np.ones((x,)), exclude_silents = True):
        def window_phon(wav):
            result = u.frame_wav(wav, winsize, winstride, window_type)
            if result.size > 0:
                return result
            else:
                return None
                
        test_matrix = []
        test_df = self.get_test_df()
        if exclude_silents:
            samples = test_df[test_df["phn"] != "h#"].sample(n=n)
        else:
            samples = test_df.sample(n=n)
                               
        for idx, s in samples.iterrows():
            result = window_phon(s["wav"])
            while result is None:
                s = test_df.sample(n=1).iloc[0]
                result = window_phon(s["wav"])
                
            test_matrix.extend(result)
        return test_matrix

    def get_train_matrix_from_wavs(self, winsize, winstride):
        pkl_file = os.path.join(self.checkpoints_folder, "train_matrix_full" + str(winsize) + "_" + str(winstride) +".pkl")
        if os.path.isfile(pkl_file):
            with open(pkl_file, 'rb') as handle:
                return pickle.load(handle)
        else: 
            train_matrix = []
            wav_files = glob.glob(os.path.join(self.dataset_folder, "TRAIN/*/*/*.wav"))
            for f in wav_files:
                wav,fr = read_wav(f)
                result = u.frame_wav(wav, winsize, winstride)
                if result.size > 0:
                    train_matrix.extend(result)   
            
            train_matrix = np.array(train_matrix)
            
            with open(pkl_file, 'wb') as handle:
                pickle.dump(train_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)    
            return train_matrix

    def get_test_matrix_from_wavs(self, winsize, winstride):
        pkl_file = os.path.join(self.checkpoints_folder, "test_matrix_full" + str(winsize) + "_" + str(winstride) +".pkl")
        if os.path.isfile(pkl_file):
            with open(pkl_file, 'rb') as handle:
                return pickle.load(handle)
        else: 
            test_matrix = []

            wav_files = glob.glob(os.path.join(self.dataset_folder, "TEST/*/*/*.wav"))
            for f in wav_files:
                wav,fr = read_wav(f)
                result = u.frame_wav(wav, winsize, winstride)
                if result.size > 0:
                    test_matrix.extend(result)   
            
            test_matrix = np.array(test_matrix)
            
            with open(pkl_file, 'wb') as handle:
                pickle.dump(test_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)    
            return test_matrix
    
    def get_train_matrix(self, winsize, winstride, window_type = lambda x:np.ones((x,))):
        pkl_file = os.path.join(self.checkpoints_folder, "train_matrix_" + str(winsize) + "_" + str(winstride) +".pkl")

        if os.path.isfile(pkl_file):
            with open(pkl_file, 'rb') as handle:
                return pickle.load(handle)
        else:    
            train_phons = self.get_train_df()
            train_matrix = []
            for index, row in train_phons.iterrows():
                result = u.frame_wav(row['wav'], winsize, winstride, window_type)
                if result.size > 0:
                    train_matrix.extend(result)                   
            train_matrix = np.array(train_matrix)
            
            with open(pkl_file, 'wb') as handle:
                pickle.dump(train_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)    
            return train_matrix
    
    def get_test_matrix(self, winsize, winstride, window_type = lambda x:np.ones((x,))):
        pkl_file = os.path.join(self.checkpoints_folder, "test_matrix_" + str(winsize) + "_" + str(winstride) + ".pkl")

        if os.path.isfile(pkl_file):
            with open(pkl_file, 'rb') as handle:
                return pickle.load(handle)
        else:    
            test_phons = self.get_test_df()
            test_matrix = []
            for index, row in test_phons.iterrows():
                result = u.frame_wav(row['wav'], winsize, winstride, window_type)
                if result.size > 0:
                    test_matrix.extend(result)                    
            test_matrix = np.array(test_matrix)
            
            with open(pkl_file, 'wb') as handle:
                pickle.dump(test_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)                  
            return test_matrix    
        
    def _read_phons(self, phn_file):
        phn_list = []
        for line in open(phn_file):
            elems = line.split(" ")
            phn_list.append({"start": int(elems[0]), "end": int(elems[1]), "phn": elems[2].rstrip()})
        return phn_list

    def _extract_phons(self, wav, phns_list):
        phns_df = pd.DataFrame(columns = ["wav", "phn", "phn_index", "category", "collapsed"])
        for phn in phns_list:
            phn_wav = wav[int(phn["start"]):int(phn["end"]+1)]
            phn_index = self.phns_all.index(phn["phn"])
            phn_label = phn["phn"]
            category = self.phns_cat_dict[phn_label]
            try:
                collapsed = self.collapse_dict[phn_label]
            except:
                collapsed = phn_label
            phns_df = phns_df.append({"wav": phn_wav, "phn": phn_label, "phn_index": phn_index,
                                      "category": category,"collapsed":collapsed}, ignore_index = True)
        return phns_df

    def _create_dict(self,phns_files):
        _phns_df = pd.DataFrame(columns = ["wav", "phn", "phn_index", "category", "collapsed"])
        for phn_file in phns_files:
            wav_file = phn_file[:-4] + "RIFF.wav"
            wav,_ = read_wav(wav_file)
            phns_list = self._read_phons(phn_file)
            _phns_df = _phns_df.append(self._extract_phons(wav, phns_list), ignore_index = True)
        return _phns_df

if __name__ == "__main__":
    die = DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")

    test_data_df = die.get_ctc_test_data()
    train_data_df = die.get_ctc_train_data()

            
#Read phonemes DataFrame, create if it doesn't exists
#phns_df_file = os.path.join(checkpoints_folder, "phns_df.pkl")
#if os.path.isfile(phns_df_file):
#    phns_df = pd.read_pickle(phns_df_file)
#else:    
#    phns_df = create_dict(phn_files)
#    pd.to_pickle(phns_df, path = phns_df_file)