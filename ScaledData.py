# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:20:41 2023

@author: tommy
"""
import numpy as np
from sklearn.preprocessing import minmax_scale

def scaledata(datain, minval, maxval):
    """ Program to scale the values of a matrix
    #     from a user specified minimun to a user specified
    #     maximum"""
    return minmax_scale(datain, feature_range=(minval,maxval))
    

if __name__ == "__main__":
    #Test the function in __main__
    x = np.random.randn(5,5)
    y = scaledata(x, 1, 5)
    print(y)
