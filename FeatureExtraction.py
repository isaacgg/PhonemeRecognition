# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:31:05 2018

@author: isaac
"""
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.signal import get_window
import python_speech_features as psf
import glob
import os
#import h5py
import pickle


def read_wav(wav_file):
    fr, wav = wavread(wav_file)
    wav = wav/np.max(np.abs(wav))
    return wav, fr  

class DataPreprocessor:
    phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
    	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
    	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]
    
    collapse_dict = {"ao":"aa", "ax":"ah", "ax-h":"ah", "axr": "er", "hv":"hh", "ix":"ih", "el":"l", "em":"m", "en":"n",
             "nx":"n", "eng":"ng","zh":"sh", "ux":"uw","pcl":"sil", "tcl":"sil", "kcl":"sil", "bcl":"sil", 
             "dcl":"sil", "gcl":"sil", "h#":"sil", "pau":"sil", "epi":"sil", "q":"-"}
    
    def __init__(self, extract_function):
        self.extract_function = extract_function
        
    def read_phons(self, phn_file):
        phn_list = []
        for line in open(phn_file):
            elems = line.split(" ")
            phn_list.append({"start": int(elems[0]), "end": int(elems[1]), "phn": elems[2].rstrip()})
        phn_list = sorted(phn_list, key=lambda x: x["start"])
        return phn_list

    def calc_norm_param(self, X, VERBOSE=False):
    	"""Assumes X to be a list of arrays (of differing sizes)"""
    	total_len = 0
    	mean_val = np.zeros(X[0].shape[1])
    	std_val = np.zeros(X[0].shape[1])
    	for obs in X:
    		obs_len = obs.shape[0]
    		mean_val += np.mean(obs,axis=0)*obs_len
    		std_val += np.std(obs, axis=0)*obs_len
    		total_len += obs_len
    	
    	mean_val /= total_len
    	std_val /= total_len
    
    	if VERBOSE:
    		print(total_len)
    		print(mean_val.shape)
    		print('  {}'.format(mean_val))
    		print(std_val.shape)
    		print('  {}'.format(std_val))
    
    	return mean_val, std_val, total_len
    
    def normalize(self, X_train, X_test):        
        def _normalize(X, mean_val, std_val):
        	for i in range(len(X)):
        		X[i] = (X[i] - mean_val)/std_val
        	return X
        
        mean_val, std_val, _ = self.calc_norm_param(X_train)
    
        X_train = _normalize(X_train, mean_val, std_val)
        X_test 	= _normalize(X_test, mean_val, std_val)
        
        return X_train, X_test
    
    def preprocess_data_single(self, DIR_PHN, ignore_SA = True):
        DIR_PHN = os.path.join(DIR_PHN, '*/*/*.PHN')
        files = glob.glob(DIR_PHN)
        
        if ignore_SA:
            files = [x for x in files if not os.path.splitext(x)[0].startswith("SA")]

        X = []
        y = []
        for phn_file in files:
            wav_file = phn_file[:-4] + "RIFF.wav"
            feats = self.extract_function(wav_file)
            feats_len = feats.shape[0]
            X.append(feats)
    
            y_val = np.zeros(feats_len) - 1
            phns_list = self.read_phons(phn_file)
            wav_len = int(phns_list[-1]["end"])
            start_ind = 0
            for phn in phns_list:
                ix = self.phonemes.index(phn["phn"])
                stop = phn["end"]
                stop_ind = int(np.round((stop)/wav_len*feats_len))
                y_val[start_ind:stop_ind] = ix
                start_ind = stop_ind
            y.append(y_val)
            
        return X, y
    
    def preprocess_data(self, TRAIN_DIR, TEST_DIR, ignore_SA = True):
        TRAIN_PHN = os.path.join(TRAIN_DIR, '*/*/*.PHN')
        TEST_PHN = os.path.join(TEST_DIR, '*/*/*.PHN')
                
        train_files = glob.glob(TRAIN_PHN)
        test_files = glob.glob(TEST_PHN)

        train_files = [x for x in train_files if not os.path.splitext(x)[0].startswith("SA")]        
        if ignore_SA:
            test_files = [x for x in test_files if not os.path.splitext(x)[0].startswith("SA")]

        
        X_train = []
        y_train = []
        for phn_file in train_files:
            wav_file = phn_file[:-4] + "RIFF.wav"
            feats = self.extract_function(wav_file)
            feats_len = feats.shape[0]
            X_train.append(feats)
    
            y_val = np.zeros(feats_len) - 1
            phns_list = self.read_phons(phn_file)
            wav_len = int(phns_list[-1]["end"])
            start_ind = 0
            for phn in phns_list:
                ix = self.phonemes.index(phn["phn"])
                stop = phn["end"]
                stop_ind = int(np.round((stop)/wav_len*feats_len))
                y_val[start_ind:stop_ind] = ix
                start_ind = stop_ind
            y_train.append(y_val)
            
        return X_train, y_train
        
        X_test = []
        y_test = []
        for phn_file in test_files:
            wav_file = phn_file[:-4] + "RIFF.wav"
            feats = self.extract_function(wav_file)
            feats_len = feats.shape[0]
            X_test.append(feats)
    
            y_val = np.zeros(feats_len) - 1
            phns_list = self.read_phons(phn_file)
            wav_len = int(phns_list[-1]["end"])
            start_ind = 0
            for phn in phns_list:
                ix = self.phonemes.index(phn["phn"])
                stop = phn["end"]
                stop_ind = int(np.round((stop)/wav_len*feats_len))
                y_val[start_ind:stop_ind] = ix
                start_ind = stop_ind
            y_test.append(y_val)
        
        #X_train, X_test = self.normalize(X_train, X_test)
        
        return X_train, y_train, X_test, y_test


from grelurbm import GReluRBM

class RBMExtractor():
    def __init__(self, winlen, winstep, n_hidden, winfunc = lambda x:np.ones((x,)), deltas=True, v=2, logsdir = "D:/TrabajoFinDeMaster/rbmrelucats/"):
        self.winlen = winlen
        self.winstep = winstep
        self.winfunc = winfunc

        self.deltas = deltas

        self.grelurbm = GReluRBM(n_visible=winlen, n_hidden=n_hidden, momentum=0, err_function = 'rmse')
        self.grelurbm.load_weights(logsdir, "weights.h5")
#        self.grelurbm.fix_hidden_bias()
        
        if v==1:
            self.feats_func = self.grelurbm.get_features
        elif v==2:
            self.feats_func = self.grelurbm.get_features_v2
        elif v==3:
            self.feats_func = self.grelurbm.get_features_v3
        elif v == 4:
            self.feats_func = self.grelurbm.get_features_v4
            
        try:
            with open(logsdir + "/mean.pkl", 'rb') as handle:
                self.mean = pickle.load(handle)
        except FileNotFoundError:
            print("mean.pkl not found!")
            self.mean = 0
        
        try:
            with open(logsdir + "/std.pkl", 'rb') as handle:
                self.std = pickle.load(handle)
        except FileNotFoundError:
            print("std.pkl not found!")
            self.std = 0.11
        
    def get_rbm(self, filename):               
        wav, fr = read_wav(filename)

        frames = psf.sigproc.framesig(sig=wav, frame_len=self.winlen, frame_step=self.winstep)
        frames = [self.winfunc*(f-self.mean)/self.std for f in frames]
        
        f_rbm = self.feats_func(frames)

        if True:
            NFFT = 512
            complex_spec = np.fft.rfft(frames, NFFT)
            power_spec =  1.0 / NFFT * np.square(np.abs(complex_spec))
            argmax = np.argmax(power_spec, axis = 1)
            argmax = argmax.reshape(len(argmax),1)
            f_rbm = np.concatenate((f_rbm, argmax), axis=1)
            

        if self.deltas:
            deltas = psf.delta(f_rbm, 1)
#            derivative = np.zeros(f_rbm.shape)
#            for i in range(1, f_rbm.shape[0]-1):
#                derivative[i, :] = f_rbm[i+1, :] - f_rbm[i-1, :]
                
            f_rbm = np.concatenate((f_rbm, deltas ), axis=1)
        
        

        return f_rbm
    
    def close_session(self):
        self.grelurbm.close_session()
        del(self.grelurbm)
    


if __name__ == "__main__":  
    import Floyd.scripts.commons.utils as utils
#    rbm = RBMExtractor(100, 50)
    
    def get_mfcc(filename):
        wav, fr = read_wav(filename)
        
        mfccs = psf.mfcc(wav,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13, nfilt=26, appendEnergy=True)
        deltas = psf.delta(mfccs, 1)
        
        out = np.concatenate((mfccs, deltas), axis=1)
        return out
#    
#    def get_mfcc_v2(filename):
#        wav, fr = read_wav(filename)
#        
#        mfccs = psf.mfcc(wav,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13, nfilt=26, appendEnergy=True)
#        
#        derivative = np.zeros(mfccs.shape)
#        for i in range(1, mfccs.shape[0]-1):
#            derivative[i, :] = mfccs[i+1, :] - mfccs[i-1, :]
#            
#        out = np.concatenate((mfccs, derivative ), axis=1)
#        return out
    
    
    #Extract RBM feats
    TRAIN_DIR = './Database/TIMIT/TRAIN'
    TEST_DIR = './Database/TIMIT/TEST'
    
    winlen = 400
    winstep = 160
    n_hidden = 50
    winfunc = "hann"
    v = 2
    
    parameters_id = str(winlen) + "_" + str(n_hidden) + "_" + winfunc
    parameters_short_id = str(winlen) + "_" + str(n_hidden)
    
#    weigths_dir = "./GRELURBM_archs/greluRBM_400_600_hann/"
    weigths_dir = "D:/TrabajoFinDeMaster/greluRBM_"+ parameters_id + "/longtrain/"
    
    rbm = RBMExtractor(winlen=winlen, winstep=winstep, n_hidden=n_hidden, winfunc = get_window(winfunc, winlen),
                       deltas=True, v=v, logsdir = weigths_dir)
    
    directory = "./PreprocessedData/RBMs_longtrain/" + winfunc + "/" + "v" + str(v) + "/" + parameters_short_id + "/"
#    directory = "./PreprocessedData/MFCCs_delta1/" + winfunc + "/"
    try:
        os.makedirs(directory)
    except FileExistsError:
        print("ya existe el directorio!!")
        print(directory)
    
    dp = DataPreprocessor(extract_function = rbm.get_rbm)
#    dp = DataPreprocessor(extract_function = get_mfcc)
    X_train, y_train = dp.preprocess_data_single(TRAIN_DIR)
    utils.save_npy(y_train, directory + "y_train.npy")
    del(y_train)
    utils.save_npy(X_train, directory + "X_train.npy")
    del(X_train)
    
    
    X_test, y_test = dp.preprocess_data_single(TEST_DIR, ignore_SA = True)
    utils.save_npy(y_test, directory + "y_test.npy")
    del(y_test)
    utils.save_npy(X_test, directory + "X_test.npy")
    del(X_test)
    
    rbm.close_session()
    del(rbm)
    del(dp)