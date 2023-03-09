# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:45:18 2023

@author: artem
"""

from ScaledData import scaledata
from ReLU import ReLU
import numpy as np
import matplotlib.pyplot as plt

"""An example of RNN learning a random target pattern
using a combination of supervised and predictive learning"""

#%% 
#RNN Parameters:

N = 1 #number of inputs
alpha = 1 #leark term
M = 200 #number of recurrent units
dt = 1 #RNN time-step

#Training Data

T = 200 #Number of time-steps
R = np.random.randn(N,T) #Random sequence
X = R #Network input
Y = X #Target

#Fixed synaptic weights

Z = np.random.randn(M,N) #Fixed random weights from input to RNN
W_in = np.random.randn(M) #Fixed random weights within RNN

#Initial conditions

B = np.random.randn(N,T) #RNN output
Bhat = np.random.randn(N,T) #Top-down output prediction
W = np.random.randn(M,N) #RNN output weights

#Time constants

tau = 0.1 #RNN time constant
tau_w = 0.1 #Supervised plasticity time constant
tau_b = 0.1 #Predictive plasticity time constant

#Plasticity parameters

gamma = 0.5 #supervised learning strength
sigma = 0.05 #predictive learning strength
num_iter = 100 #number of training iterations


#%%
#Generate RNN activity:
    
I = Z@X
K = np.zeros((M,T))
K[:,0] = I[:,0]
T_dt = int(T/dt)

for t in range(1,T_dt):
    dK = -alpha*K[:,t-1] + ReLU(K[:,t-1]*W_in) +I[:,t]+1
    K[:,t] += tau*dK
    
#Scaling the data to help with plotting
Ks = scaledata(K, 0, (1/(M**2)))

#%%


















































