# MSc_thesis
This repository contains all code and data to reproduce results for my thesis for MSc Econometrics. Each file has a short description of its functionality. Important: the source file with all data for all European countries was too big to upload to github. Therefore I have only uploaded the log-returns. I can send the source file on request via WeTransfer.

## Data preparation
Contains the exploratory data analysis file. This file was used to clean the data and produce log-returns that in turn would be used by the .R files for volatility modeling.

## 1. Electricity price dataset
Contains all the log-returns needed for this research, and the dates that accompany the log-return data. 

## 2. GARCH modeling
The file contains the most important scripts for the conditional volatilit models. R is used to model the electricity price data with the conditional volatility models, ARCH, GARCH, E-GARCH, GJR-GARCH and MS-GARCH. The main reason is the library called MS-GARCH which is implemented in R and not in python. R has extensive libraries on conditional volatility models useful for this thesis. The main file that is used. tocall all necessary functions to produce forecasts and features is called: GARCH_models_nstep.R. This file is used to produce forecasts and features. Important: forecasts are out-of-sample for the test set. Features are the forecasts also for the in-sample dataset, because these features are needed as input for the hybrid models that need in-sample forecasts as training data. The folder 1_EP contains the predictions and features for all horizons for the electricity price data for all countries. The folder 2_Simulated_sequences contains all the features used for the hybrid modeling (predictions not needed for the simulation study, because only LSTM and hybrid models are compared). It also contains al the simulated sequences. 

## 3. Figures
Contains several plots of the forecasts of the GARCH family and LSTM models.

## 4. LSTM predictions
Fitting all LSTM and hybrid models to the data is very time intensive. Due to large dat available, and the goal to outperform the naive predictions, a lot of parameters and nodes are needed to improve forecasting performance. Therefore all predictions of the LSTM's are included. The files are named as follows: LSTM_preds_arch_ged_12 are the predictions of the hybrid LSTM model that got the realized volatility and the forecasts of the ARCH(1) model as added feature as input and was used to produce 12-step ahead forecasts [LSTM_preds_model_distr_nahead.csv]. The folder name describes the country for which these predictions were made. 
(Sidenote, this was done on a MAC with 8 GB RAM). 

## EP_modeling_file.ipynb
Calls all the functions from model_functions.py to produce forecasts and forecast results for the electricity price data in Spain. Via plots and matrices that can directly be exported to LaTex. It does the visualizations of the results, fit ARMA models on residuals and all other manipulations mentioned in the report. 

## Model_functions.py
Contains all functions used to produce the results of the forecasting study. Each function has a detailed description of its functionality.

## Simulation_study.ipynb
This file produces all forecast errors calculated for the simulated sequences. These sequences were produced in R using and all features were produced by Sim_GARCH_models_nstep.R


