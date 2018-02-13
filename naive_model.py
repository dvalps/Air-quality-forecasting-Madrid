"""
Benchmark models: Naive model.
"""

import numpy as np
import pandas as pd

from evaluate_forecast import evaluate_forecast

# DEFINE THE FORECASTING HORIZON
k_hrs = 1

##################################################################
y2013 = pd.read_csv('data_frames/data2013.csv')
y2014 = pd.read_csv('data_frames/data2014.csv')
y2015 = pd.read_csv('data_frames/data2015.csv')


# the first column is not a date object
y2015['date'] = y2015['date'].apply(pd.to_datetime)
y2014['date'] = y2014['date'].apply(pd.to_datetime)
y2013['date'] = y2013['date'].apply(pd.to_datetime)

# Take the NO2 values from the data frame
no2_2015 = y2015.values[:,1:25]
no2_2014 = y2014.values[:,1:25]
no2_2013 = y2013.values[:,1:25]

##############################################################
#Combine NO2 data into one matrix
real_no2 = np.vstack([no2_2013, no2_2014, no2_2015])

no2_ts = real_no2.reshape(-1,1) # flatten it into a vector

###############################################################
# NAIVE PREDICTOR 1 - tries to predict t-12, the forecast is no2 at time t

# pred1 predicts taking the value at time t, first available prediction is at time t
pred1 = no2_ts[:-k_hrs]
y1_real = no2_ts[k_hrs:]


######################################################################
# TEST ERROR ###############################
rmse1, mae1, ia1, mb1, pears1 = evaluate_forecast(y1_real, pred1, normalize = 0)
errors1 = np.array([rmse1, mae1, ia1, mb1, pears1])
rmse1, mae1, ia1, mb1, pears1 = evaluate_forecast(y1_real, pred1, normalize = 1)
errors1_norm = np.array([rmse1, mae1, ia1, mb1, pears1])
print("Errors - Naive model 1: ")
print(errors1)
