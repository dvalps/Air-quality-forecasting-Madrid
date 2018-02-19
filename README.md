# Air-quality-forecasting-Madrid

System for forecasting hourly concentrations of Nitrogen dioxide in Madrid.

Learners developed for forecasting air concentrations in the station of Plaza de España, Madrid, Spain.

---------------------------------------------------------------------------------
FILE DESCRIPTION
---------------------------------------------------------------------------------
- data_frames -     contains data frames with the data samples with all the time series used in creating features
- train_examples -  matrices of training examples, extracted from data frames, one for each forecasting horizon: 1, 2,
                    6, 12, 24, and 48 hours
- targets -         vectors with target values, one for each forecasting horizon: 1, 2, 6, 12, 24, and 48 hours


--------------------------------------------------------------------------------------
Implemented learners are presented in the following files. The hyperparameters are set to the values obtained through the tunning process (can be changed if want to experiment with it).

NOTE: It is needed to change the parameter 'k_hrs' (at the beginning of the implementation file .py) to the value of the forecasting horizon that wants to be used: 1, 2, 6, 12, 24, 48.

- naive_model.py - implementation of a naive predictor that takes the current value to the predict a future one
- LinearRegression.py - implementation of a linear regression model
- SVM.py - implementation of a support vector machine model
- regression_forest.py - implementation of a regression forest model
- FNN.py - implementation of (shallow and deep) feedforward neural networks (set the desired depth, up to 3 hidden layers)
- lstm.py - implementation of an LSTM neural network


P. S. Work in progress. Moving towards an operational deep learning forecasting system. The code uploaded so far serves mainly for demostration purposes.
