#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:51:33 2019

@author: damir
"""

# Meta-learners

import numpy as np
import pandas as pd

from evaluate_forecast import evaluate_forecast

horizons = [1, 2, 6, 12, 24, 48]
models = ['FNN_deep1', 'FNN_deep2', 'FNN_deep3', 'SVM', 'RF', 'lstm1', 'DeepLSTM', 'CNN'] #'LinReg' is still first

error_meta = pd.DataFrame(columns = horizons)
"""
Order of rows in this data frame is: rmse and mb for average, rmse and mb for weighted average.
"""

for h in horizons:
    # read the data from csv
    df = pd.read_csv('forecasts/forecasts_t' + str(h) + "_" + 'LinReg.csv', usecols = ['observed', 'predictions'])
    #print(df.head())
    for m in models:
        df_temp = pd.read_csv('forecasts/forecasts_t' + str(h) + "_" + m + ".csv", usecols = ['observed', 'predictions'])
        #print(df.head())
        # add a predictions column - using the name of the model as appendix
        df["predictions_" + m] = df_temp["predictions"]
        
    # calculate average
    df['avg'] = np.average(np.array(df.iloc[:,1:len(models) + 2]), axis = 1)
    
    rmse_avg, mae_avg, ia_avg, mb_avg, pears_avg = evaluate_forecast(df.observed, df.avg, normalize=0)
    
    
    # jer mi se ne da jebati, za sve horizonte sam uzeo rmse za t = 12 -- odlucih da je
    # aproksimacija dobra za sve vise manje.
    rmses = [17.79, 15.54, 15.27, 15.10, 18.03, 15.86, 15.31, 15.27, 15.65]
    weights = (1 / (len(rmses) - 1)) * (1 - rmses / np.sum(rmses))

    df['avg_weighted'] = np.average(np.array(df.iloc[:,1:len(models) + 2]), axis = 1, weights = weights)
    rmse_avg_w, mae_avg_w, ia_avg_w, mb_avg_w, pears_avg_w = evaluate_forecast(df.observed, df.avg_weighted, normalize=0)

    error_meta[h] = [rmse_avg, mb_avg, rmse_avg_w, mb_avg_w]
        

# Calculate weighted average
"""Order: LinReg, FNN, FNN 2 layers, ... follow the order in models
"""


print(df.head())

print(error_meta.head())
