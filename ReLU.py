# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:12:45 2023

@author: tommy
"""

import numpy as np

# def ReLU(x):
#     T = len(x)
#     y = np.zeros(T,1)
#     f = np.argwhere(x)
    
def ReLU(x):
    return np.maximum(x, 0, x)

if __name__ == '__main__':
    #Test the function
    x = np.random.randn(4,5)
    print(x)
    ReLU(x)
