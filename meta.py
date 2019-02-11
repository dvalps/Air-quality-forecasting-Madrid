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
models = ['FNN_deep1', 'FNN_deep2', 'FNN_deep3', 'SVM', 'RF', 'lstm1'] #'LinReg',


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
df['avg'] = df.mean(axis = 1, skipna=True)
print(df.head())

rmse_avg, mae_avg, ia_avg, mb_avg, pears_avg = evaluate_forecast(df.observed, df.avg, normalize=0)

# Calculate weighted average
pass
"""
TO DO
1 - (WEIGHT / WEIGHT.sum()), weights = RMSE_test
"""

