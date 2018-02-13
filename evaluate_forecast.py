#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:57:46 2017

@author: dvalput
"""

from sklearn.metrics import mean_squared_error
import numpy as np

# Evaluation of NO2 predictions, takes vector o and p.

def evaluate_forecast(o, p, normalize=1):
    # o - vector of real, observed NO2 values
    # p - vector of forecasts for NO2 values
    # They have to be DENORMALIZED and of the same shape.
    
    # If normalized = 1, then normalized values of RMSE, MAE AND MB are returned.
    
    # Returns: RMSE, MAE, IA, MB (mean bias.)
    
    if not normalize:
        rmse = float(mean_squared_error(o, p)**0.5)
        
        mae = float(sum(abs(o - p)) / len(o))
        
        # mean (forecasting) bias
        mb = float(sum(p - o) / len(p))
        
    nom = sum((o - p)**2)
    o_mean = np.mean(o)
    denom = sum((abs(p - o_mean) + abs(o - o_mean))**2)
    ia = float(1 - nom/denom)
    
    # Pearson product moment correlation coefficient
    p_mean = float(np.mean(p))
    o_mean = float(np.mean(o))
    p_dev = float(np.std(p))
    o_dev = float(np.std(o))
    nominator = float(np.mean( (p-p_mean) * (o - o_mean) ))
    pears = (nominator / (p_dev * o_dev))

    
    if normalize:
        rmse = float( (sum( ((o - p) / o)**2) / len(o)) ** 0.5 )
        
        mae = float(sum(abs((o - p) / o)) / len(o))
        
        mb = float(sum( (p - o ) / o) / len(p))
    
    return(rmse, mae, ia, mb, pears)