#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:34:17 2018

@author: dvalput
"""
import numpy as np

def normalize_by_columns_maxmin(X, y):
    # X - matrix of training examples
    # y - vector of targets
    
    # Normalizes by using the maximum value in each column
    
    X_norm = np.empty(X.shape)
    
    X_max = np.amax(X, axis = 0) # maximum by columns
    X_min = np.amin(X, axis = 0)
    
    for i in range(0, X.shape[1]): # normalize by columns
        X_norm[:,i] = (X[:,i] - X_min[i]) / (X_max[i] - X_min[i])
    
    y_max = np.amax(y)
    y_min = np.amin(y)
    y_norm = (y-  y_min) / (y_max - y_min)
    
    return (X_norm, y_norm, y_max, y_min) # I need to return the max and min of NO2 for denormalization


def denormalize_maxmin(y, y_max, y_min):
    
    y_denorm = y * (y_max - y_min) + y_min
    
    return y_denorm