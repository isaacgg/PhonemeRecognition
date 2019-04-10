# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:13:46 2018

@author: isaac
"""

import torch
import numpy as np

class HiddenMarkovModel():
    
    dtype = torch.float
    def __init__(self, n_states, n_obs):
        """ STORE VARIABLES """
        self.n_states = n_states
        self.n_obs = n_obs
        
        """ INITIALIZE MATRICES """ 
        #TODO: PI_end should only be possible in las state
        #TODO: Pi should be [1 0 0]
        #TODO: A should only transit to forward states
        
        self.Pi = torch.tensor(self._init_probs((self.n_states)), dtype = self.dtype) #initial state probs
        self.A = torch.tensor(self._init_probs((self.n_states, self.n_states)), dtype = self.dtype) #transition state probs
        self.pi_final = torch.tensor(self._init_probs((self.n_states)), dtype = self.dtype) #end probs
        self.B = torch.tensor(self._init_probs((self.n_states, self.n_obs)), dtype = self.dtype) #emision probs

    """ ALGORITHMS """    
    def forward(self, obs):
        fwd = torch.zeros(self.n_states, len(obs))
        fwd[:,0] = self.pi*self.B[:,obs[0]]

        for ix, ob in enumerate(obs[1:]):
            prev_fwd = fwd[:, ix].clone()
            fwd[:, ix+1] = torch.mm(prev_fwd, self.A[:,:]*self.B[:,ob])
#            for state in range(self.n_states):
#                fwd[state, ix+1] = torch.dot(prev_fwd, self.A[:,state]*self.B[state,ob])
        
        if self.pi_final is not None:
            fwd_f = torch.sum(fwd[:,-1]*self.pi_final)
        else:
            fwd_f = torch.sum(fwd[:,-1])
            
        return fwd_f, fwd
    
    def backward(self, obs):
        if self.pi_final is not None:
            back = self.pi_final
        else:
            back = torch.ones(self.n_states)
            
        for ob in reversed(obs[1:]):
            prev_back = back.clone()
            for state in range(self.n_states):
                back[state] = torch.dot(self.A[state,:], prev_back*self.B[:,ob])
        
        bck_f = torch.dot(self.pi, back*self.B[:,obs[0]])
        
        return bck_f, back
    
    def viterbi(self, obs):
        vit = torch.zeros(self.n_states, dtype = self.dtype)
        path = torch.zeros((1, len(obs)), dtype = torch.int)
        
        vit = self.pi*self.B[:,obs[0]] #[1,4].*[1,4] = [1,4]
        
        for idx_ob, ob in enumerate(obs[1:]):
            vit_prev = vit.clone()
            for state in range(self.n_states):
                vit[state] = torch.max(vit_prev*self.A[:,state])*self.B[state,ob]
            path[0,idx_ob+1] = torch.argmax(vit)
            
        if self.pi_final is not None:
            for state in range(self.n_states):
                vit[state] = torch.max(vit_prev*self.pi_final[state])
            path[0,idx_ob+2] = torch.argmax(vit)
        
        return path, vit
    
    def forward_backward(self, obs):
        _,fwd = self.forward(obs)
        back, back_end = self.backwards(obs)
        
        
        gamma = fwd*back/back_end
        
        return gamma
    
    """ UTILS """
    def _init_probs(self, size):
        arr = np.exp(np.abs(np.random.normal(loc=2,size=size)))
        div = np.sum(arr.T, axis = 0, keepdims = True)
        norm = arr/div.T
        return norm
    
    def load(self, A, B, Pi, Pi_end):
        self.A = torch.tensor(A, dtype = self.dtype)
        self.B = torch.tensor(B, dtype = self.dtype)
        self.pi = torch.tensor(Pi, dtype = self.dtype)
        if pi_final is not None:
            self.pi_final = torch.tensor(Pi_end, dtype = self.dtype)
        else:
            self.pi_final = None
            
B = np.array([[0.3, 0.1, 0.6],[0.4,0.3,0.3]]) 
pi = np.array([0.4, 0.6])
A = np.array([[0.3,0.7],[0.4,0.6]])
pi_final = None
obs = [2,1,0,0,1,2,2,2]

hmm = HiddenMarkovModel(2, 3)
hmm.load(A,B,pi,pi_final)
forward = hmm.forward(obs)
forward[0].numpy()
forward[1].numpy().T
viterbi = hmm.viterbi(obs).numpy()
backward = hmm.backward(obs).numpy()
