#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:51:33 2019

@author: damir
"""

# Meta-learners

import numpy as np
import pandas as pd

horizons = [1, 2, 6, 12, 24, 48]
models = ['FNN_deep1', 'FNN_deep2', 'FNN_deep3', 'SVM', 'LinReg', 'RF', 'lstm1']

# read the data from csv
df = pd.read_csv('forecasts_t' + str(1) + "_" + 'LinReg')
print(df.head())
for h in horizons[0:2]:
    for m in models:
        df_temp = pd.read_csv('forecasts_t' + str(h) + "_" + m)