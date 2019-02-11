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
models = ['FNN_deep1', 'FNN_deep2', 'FNN_deep3', 'SVM', 'RF', 'lstm1'] #'LinReg' is still first


for h in horizons[0:1]:
    # read the data from csv
    df = pd.read_csv('forecasts/forecasts_t' + str(h) + "_" + 'LinReg.csv', usecols = ['observed', 'predictions'])
    #print(df.head())
    for m in models:
        df_temp = pd.read_csv('forecasts/forecasts_t' + str(h) + "_" + m + ".csv", usecols = ['observed', 'predictions'])
        #print(df.head())
        # add a predictions column - using the name of the model as appendix
        df["predictions_" + m] = df_temp["predictions"]
        


# calculate average
df['avg'] = df.mean(axis = 1, skipna=True)
print(df.head())

rmse_avg, mae_avg, ia_avg, mb_avg, pears_avg = evaluate_forecast(df.observed, df.avg, normalize=0)

# Calculate weighted average
"""Order: LinReg, FNN, FNN 2 layers, ... follow the order in models
"""

rmses = [5, 10, 20, 15, 3, 5, 8]
weights = (1 / (len(rmses) - 1)) * (1 - rmses / np.sum(rmses))

df['avg_weighted'] = np.average(np.array(df.iloc[:,1:len(models) + 2]), axis = 1, weights = weights)
print(df.head())

rmse_avg_w, mae_avg_w, ia_avg_w, mb_avg_w, pears_avg_w = evaluate_forecast(df.observed, df.avg_weighted, normalize=0)

