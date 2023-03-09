# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:45:18 2023

@author: artem
"""

from ScaledData import scaledata
from ReLU import ReLU
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

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
    dK = -alpha*K[:,t-1] + ReLU(((K[:,t-1].T * W_in).T)) + I[:,t] + 1
    K[:,t] += tau*dK
    
#Scaling the data to help with plotting
Ks = scaledata(K, 0, (1/(M**2)))

#%%
#Main loop for synaptic plasticity
sup_err = np.zeros((num_iter,1))
pred_err = np.zeros((num_iter,1))
dBhat = np.zeros((1,200))

for i in range(0,num_iter):
    #combined supervised and predictive learning rule
    dW = (np.linalg.pinv(Ks.T)) @ (gamma*(Y-B)).T + (np.linalg.pinv(Ks.T)) @ (-sigma*(B-Bhat)).T
    W += (tau_w*dW)
    
    #   #update top-down prediction
    dBhat = -Bhat + B
    Bhat += tau_b * dBhat
    
    #   #network output
    B = (Ks.T @ W).T
    
    # #supervised and predictive error
    sup_err[i] = np.mean(np.power((B-Y),2))
    pred_err[i] = np.mean(np.power((Bhat-B),2))
    
#%%  
#Plot results 
Bplot = scaledata(B.T, 0, 1)
Xplot = scaledata(Y.T, 0, 1)

plt.subplot(2,2,(1,2))
plt.plot(Bplot, linewidth = 12)
plt.plot(Xplot, color = ([217/255,83/255,24/255]), linewidth = 6)
plt.axis('off')
# plt.set

plt.subplot(2,2,3)
plt.plot(pred_err, color = 'k', linewidth = 5)
plt.rcParams.update({'font.size': 18})
plt.xlabel('time-steps')
plt.ylabel('MSE')
plt.title('Supervised Error')


plt.subplot(2,2,4)
plt.plot(pred_err, color = 'k', linewidth = 5)
plt.rcParams.update({'font.size': 18})
plt.xlabel('time-steps')
plt.ylabel('MSE')
plt.title('Predictive Error')
plt.tight_layout()

plt.show()








































