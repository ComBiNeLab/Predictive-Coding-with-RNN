# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:20:41 2023

@author: artem
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scaledata(datain, minval, maxval):
    """ Program to scale the values of a matrix
    #     from a user specified minimun to a user specified
    #     maximum"""
    # return minmax_scale(datain, feature_range=(minval,maxval))
    # dataout = datain - np.min(datain)
    # max1, min1 = np.amax(datain), np.amin(datain)
    # ran = max1-min1
    # dataout = (dataout/ran)*(maxval-minval)
    # dataout += minval
    scaler = MinMaxScaler(feature_range=(minval,maxval), copy = False)
    scaler.fit(datain)
    scaler.data_max_
    return scaler.transform(datain)
    

if __name__ == "__main__":
    #Test the function in __main__
    x1 = np.random.randn(5,5)
    y1 = scaledata(x1, 0, 1)
    print(y1)
