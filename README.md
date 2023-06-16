# MSc_thesis
This repository contains all code and data to reproduce results for my thesis for MSc Econometrics. Each file has a short description of its functionality. Important: the source file with all data for all European countries was too big to upload to github. Therefore I have only uploaded the log-returns. I can send the source file on request via WeTransfer.

# Data preparation

# 1. Electricity price dataset
Contains all data needede for this research. 

# 2. GARCH modeling
R is used to model the electricity price data with the conditional volatility models, ARCH, GARCH, E-GARCH, GJR-GARCH and MS-GARCH. The main reason is the library called MS-GARCH which is implemented in R and not in python. R has extensive libraries on conditional volatility models useful for this thesis.

# 3. Figures

# 4. LSTM predictions

### R ###

EP_GARCH_models_nstep.R:

### Python ###
Python is used for its extensive library's for neural network modeling such as Keras and TensorFlow. Our study uses the TensorFlow library that allows for more flexibility than the Keras library.

Model_functions.py:
All functions used to produce forecasts using LSTM and hybrid models, each function has detailed description of its inputs and outputs.

EP_modeling_file.ipynb:
The notebook that calls the functions written in model_function.py to produce forecasts. It imports 2 important datasets generated by the EP_GARCH_models_nstep.R file. In-sample forecasts are used as features to train the hybrid LSTM models. The out-of-sample forecasts are used to calculate forecast errors and used by the hybrid LSTM top prodcue forecasts.

### Files ###
Because the fitting of the models and producing of the forecasts is very time intensive, especially the entire simulation study and applied study can take over 48 hours, mainly due to fitting the LSTM and hybrid models and producing multiple forecasts for multiple horizons. (Sidenote, this was done on a MAC with 8 GB RAM). All CSV's needed for my analysis are included to speed up the process. 

LSTM_preds/EP/:
This folder contains all the predictions of the LSTM and hybrid models. For example: lstm_preds_arch_std_12.csv contains the predictions of the hybrid LSTM-ARCH model assuming a student-t distributions, containing 12-step ahead forecasts.

Simulated_sequences/:
This folder contains all the simulated time series sequences, the prediction of the volatility of these sequences and the 1, 12 and 24 step ahead forecast for each point in the time series that is used as a feature for the hybrid LSTM modeling. That is why the files are called: features.



