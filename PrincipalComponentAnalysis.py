# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:48:00 2018

@author: isaac
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as pca_skl
import numpy as np
import pandas as pd

class PCA_from_df():
    def __init__(self, pca_dims):
        self.pca_dims = pca_dims 
        
    def _prepare_pca_data(self, df, algth = "fbank"):
        feats = []
        idxs = []
        for ix, row in df[df["algth"] == algth].iterrows():
            num_windows = row["feats"].shape[0]
            ixs = np.repeat(row["phn"], num_windows)
    
            feats.extend(row["feats"])
            idxs.extend(ixs)
    
        return np.array(feats, dtype = float), idxs
    
    def _normalize_data(self, data):
        return StandardScaler().fit_transform(data)
    
    def compute_pca(self, df, algth = "fbank"):
        pca_data, indexs = self._prepare_pca_data(df, algth)
        norm_data = self._normalize_data(pca_data)
        
        pca = pca_skl(n_components=self.pca_dims)
        principalComponents = pca.fit_transform(norm_data)
        principalDf = pd.DataFrame(data = principalComponents,
                                   columns = ["pca" + str(i) for i in range(self.pca_dims)])
        
        finalDf = pd.concat([principalDf, pd.DataFrame(indexs, columns = ["target"])], axis = 1)
        return finalDf
    
class PCA():
    def _normalize_data(self, data):
        return StandardScaler().fit_transform(data)

    def compute_pca(self, feats, out_dims = 3):
        norm_data = self._normalize_data(feats)
        
        pca = pca_skl(n_components=out_dims)
        principalComponents = pca.fit_transform(norm_data)
        
        return principalComponents