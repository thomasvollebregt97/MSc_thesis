# Modeling multiple step ahead forecasts
library(fGarch)
library(dplyr)
library(tseries)
library(rugarch)
library(MSGARCH) 
source("forecast_eval.R")
source("forecast_eval_arch.R")
source("forecast_eval_ms.R")
source("forecast_eval_ms_plus.R") #For generating 1 extra filtered observation
source("rolling_forecast_ms.R")
source("rolling_forecast_ms_4f.R")
source("filtered_ms.R")
source("produce_forecast_and_filter_ms.R")
source("generate_matrix_results.R")

# Generate 1 step ahead forecasts
source("produce_1_step_feature.R")
source("produce_1_step_feature_arch.R")

# Generate n step ahead forecasts
source("generate_n_step_forecasts_arch.R")
source("generate_n_step_forecasts.R")

# Produce vector forecasts for entire set, for feature engineering
source("produce_vector_forecasts_arch.R")
source("produce_vector_forecasts.R")

# Produce predictions using MS-GARCH (no features!)
source("produce_ms_nstep_forecasts.R")
source("produce_ms_nstep_forecasts_new.R")
source("produce_ms_nstep_features_new.R")

" This file turn 1 sequence of log-returns into 4 matrices. 2 matrices with predictions 
using 5 models with 3 different distribution using 12 and 24-step ahead forecast horizons,
and 2 matrices with for each log_return an associated h-step ahead volatility forecast
(12 and 24, respectively). To complete this file we could look at 1-step ahead features,
the forecasting for 1-step ahead is done in file: GARCH_models_1step.R. We can repeat this
script for all the simulated sequences if we want to. We can also repeat it for the 
commodities, but we need to tune the hyperparameters as the data is much scarcer."


#### Importing the data ####
log_returns = read.csv("log_returns_portugal_new.csv", header = TRUE)
log_returns = log_returns$Price..EUR.MWhe.

plot(log_returns)

# Replace values lower than -5 with -5
log_returns <- pmax(log_returns, -5)
# Replace values greater than 5 with 5
log_returns <- pmin(log_returns, 5)
plot(log_returns)

# Splitting the data into train and test
h = 20000 #(~30%) Forecast horizon, choose a multiple of the refit.every variable
log_returns_train = log_returns[1:(length(log_returns)-h)] 
log_returns_test = log_returns[(length(log_returns)-h+1):(length(log_returns))]

# RV calculation
horizon = 50
rv = vector(mode = "numeric", length = length(log_returns) - (horizon-1))

#Shifting the window 1-step at a time
for (i in 1:(length(rv))) {
  rv[i] = sqrt(var(log_returns[(i):((horizon-1)+i)])) #Realized volatility is set equal to the variance of the subset of log-returns
}


################ Predictions: N-step ahead forecasts GARCH-type models ##################

" Here we produce the predictions for the out-of-sample set, h determines the amount that
is out of sample, here we set it at 20.000. H1 is the first horizon set at 12 (half a day).
H2 is set at 24 (full day)."

# 1-step ahead
" These are done via functions: GARCH_models_1step.R, the predictions are done for all 
models, and 4 1-step ahead features are calculated via this R script."
h = 20000
ref = 10000

arch_norm_1 = forecast_eval_arch(log_returns, "sGARCH", "norm", h = h, refit_every = ref)[,1]
arch_std_1 = forecast_eval_arch(log_returns, "sGARCH", "std", h = h, refit_every = ref)[,1]
arch_ged_1 = forecast_eval_arch(log_returns, "sGARCH", "ged", h = h, refit_every = ref)[,1]

garch_norm_1 = forecast_eval(log_returns, "sGARCH", "norm", h = h, refit_every = ref)[,1]
garch_std_1 = forecast_eval(log_returns, "sGARCH", "std", h = h, refit_every = ref)[,1]
garch_ged_1 = forecast_eval(log_returns, "sGARCH", "ged", h = h, refit_every = ref)[,1]

egarch_norm_1 = forecast_eval(log_returns, "eGARCH", "norm", h = h, refit_every = ref)[,1]
egarch_std_1 = forecast_eval(log_returns, "eGARCH", "std", h = h, refit_every = ref)[,1]
egarch_ged_1 = forecast_eval(log_returns, "eGARCH", "ged", h = h, refit_every = ref)[,1]

gjrgarch_norm_1 = forecast_eval(log_returns, "gjrGARCH", "norm", h = h, refit_every = ref)[,1]
gjrgarch_std_1 = forecast_eval(log_returns, "gjrGARCH", "std", h = h, refit_every = ref)[,1]
gjrgarch_ged_1 = forecast_eval(log_returns, "gjrGARCH", "ged", h = h, refit_every = ref)[,1]

#MSGARCH 
msgarch_norm_1 = produce_ms_nstep_forecasts_new(n_ahead = 1, "norm", log_returns_train, log_returns_test, h)
msgarch_std_1 = produce_ms_nstep_forecasts_new(n_ahead = 1, "std", log_returns_train, log_returns_test, h)
msgarch_ged_1 = produce_ms_nstep_forecasts_new(n_ahead = 1, "ged", log_returns_train, log_returns_test, h)


# 12-step ahead
h1 = 12
arch_norm_12 = generate_n_step_forecasts_arch("sGARCH", "norm", h, log_returns, h1)
arch_std_12 = generate_n_step_forecasts_arch("sGARCH", "std", h, log_returns, h1)
arch_ged_12 = generate_n_step_forecasts_arch("sGARCH", "ged", h, log_returns, h1)

garch_norm_12 = generate_n_step_forecasts("sGARCH", "norm", h, log_returns, h1)
garch_std_12 = generate_n_step_forecasts("sGARCH", "std", h, log_returns, h1)
garch_ged_12 = generate_n_step_forecasts("sGARCH", "ged", h, log_returns, h1)

egarch_norm_12 = generate_n_step_forecasts("eGARCH", "norm", h, log_returns, h1)
egarch_std_12 = generate_n_step_forecasts("eGARCH", "std", h, log_returns, h1)
egarch_ged_12 = generate_n_step_forecasts("eGARCH", "ged", h, log_returns, h1)

gjrgarch_norm_12 = generate_n_step_forecasts("gjrGARCH", "norm", h, log_returns, h1)
gjrgarch_std_12 = generate_n_step_forecasts("gjrGARCH", "std", h, log_returns, h1)
gjrgarch_ged_12 = generate_n_step_forecasts("gjrGARCH", "ged", h, log_returns, h1)

# New
msgarch_norm_12 = produce_ms_nstep_forecasts_new(n_ahead = h1, "norm", log_returns_train, log_returns_test, h)
msgarch_std_12 = produce_ms_nstep_forecasts_new(n_ahead = h1, "std", log_returns_train, log_returns_test, h)
msgarch_ged_12 = produce_ms_nstep_forecasts_new(n_ahead = h1, "ged", log_returns_train, log_returns_test, h)

# 24-step ahead
h2 = 24

arch_norm_24 = generate_n_step_forecasts_arch("sGARCH", "norm", h, log_returns, 24)
arch_std_24 = generate_n_step_forecasts_arch("sGARCH", "std", h, log_returns, 24)
arch_ged_24 = generate_n_step_forecasts_arch("sGARCH", "ged", h, log_returns, 24)

garch_norm_24 = generate_n_step_forecasts("sGARCH", "norm", h, log_returns, 24)
garch_std_24 = generate_n_step_forecasts("sGARCH", "std", h, log_returns, 24)
garch_ged_24 = generate_n_step_forecasts("sGARCH", "ged", h, log_returns, 24)

egarch_norm_24 = generate_n_step_forecasts("eGARCH", "norm", h, log_returns, 24)
egarch_std_24 = generate_n_step_forecasts("eGARCH", "std", h, log_returns, 24)
egarch_ged_24 = generate_n_step_forecasts("eGARCH", "ged", h, log_returns, 24)

gjrgarch_norm_24 = generate_n_step_forecasts("gjrGARCH", "norm", h, log_returns, 24)
gjrgarch_std_24 = generate_n_step_forecasts("gjrGARCH", "std", h, log_returns, 24)
gjrgarch_ged_24 = generate_n_step_forecasts("gjrGARCH", "ged", h, log_returns, 24)

# New
msgarch_norm_24 = produce_ms_nstep_forecasts_new(n_ahead = 24, "norm", log_returns_train, log_returns_test, h)
msgarch_std_24 = produce_ms_nstep_forecasts_new(n_ahead = 24, "std", log_returns_train, log_returns_test, h)
msgarch_ged_24 = produce_ms_nstep_forecasts_new(n_ahead = 24, "ged", log_returns_train, log_returns_test, h)

# h = 1
s = 5 # Amount of models
l = 3 # Amount of distributions
EP_prediction_matrix_1 = matrix(nrow = length(arch_norm_1), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_prediction_matrix_1) = col_names_pred

#ARCH
EP_prediction_matrix_1[,1] = arch_norm_1
EP_prediction_matrix_1[,2] = arch_std_1
EP_prediction_matrix_1[,3] = arch_ged_1
#GARCH
EP_prediction_matrix_1[,4] = garch_norm_1
EP_prediction_matrix_1[,5] = garch_std_1
EP_prediction_matrix_1[,6] = garch_ged_1
#E-GARCH
EP_prediction_matrix_1[,7] = egarch_norm_1
EP_prediction_matrix_1[,8] = egarch_std_1
EP_prediction_matrix_1[,9] = egarch_ged_1
#GJR-GARCH
EP_prediction_matrix_1[,10] = gjrgarch_norm_1
EP_prediction_matrix_1[,11] = gjrgarch_std_1
EP_prediction_matrix_1[,12] = gjrgarch_ged_1
#MS-GARCH
EP_prediction_matrix_1[,13] = msgarch_norm_1
EP_prediction_matrix_1[,14] = msgarch_std_1
EP_prediction_matrix_1[,15] = msgarch_ged_1

# h = 12
s = 5 # Amount of models
l = 3 # Amount of distributions
EP_prediction_matrix_12 = matrix(nrow = length(arch_norm_12), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_prediction_matrix_12) = col_names_pred

#ARCH
EP_prediction_matrix_12[,1] = arch_norm_12
EP_prediction_matrix_12[,2] = arch_std_12
EP_prediction_matrix_12[,3] = arch_ged_12
#GARCH
EP_prediction_matrix_12[,4] = garch_norm_12
EP_prediction_matrix_12[,5] = garch_std_12
EP_prediction_matrix_12[,6] = garch_ged_12
#E-GARCH
EP_prediction_matrix_12[,7] = egarch_norm_12
EP_prediction_matrix_12[,8] = egarch_std_12
EP_prediction_matrix_12[,9] = egarch_ged_12
#GJR-GARCH
EP_prediction_matrix_12[,10] = gjrgarch_norm_12
EP_prediction_matrix_12[,11] = gjrgarch_std_12
EP_prediction_matrix_12[,12] = gjrgarch_ged_12
#MS-GARCH
EP_prediction_matrix_12[,13] = msgarch_norm_12
EP_prediction_matrix_12[,14] = msgarch_std_12
EP_prediction_matrix_12[,15] = msgarch_ged_12

# h = 24
s = 5 # Amount of models
l = 3 # Amount of distributions

EP_prediction_matrix_24 = matrix(nrow = length(arch_norm_24), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_prediction_matrix_24) = col_names_pred

#ARCH
EP_prediction_matrix_24[,1] = arch_norm_24
EP_prediction_matrix_24[,2] = arch_std_24
EP_prediction_matrix_24[,3] = arch_ged_24
#GARCH
EP_prediction_matrix_24[,4] = garch_norm_24
EP_prediction_matrix_24[,5] = garch_std_24
EP_prediction_matrix_24[,6] = garch_ged_24
#E-GARCH
EP_prediction_matrix_24[,7] = egarch_norm_24
EP_prediction_matrix_24[,8] = egarch_std_24
EP_prediction_matrix_24[,9] = egarch_ged_24
#GJR-GARCH
EP_prediction_matrix_24[,10] = gjrgarch_norm_24
EP_prediction_matrix_24[,11] = gjrgarch_std_24
EP_prediction_matrix_24[,12] = gjrgarch_ged_24
#MS-GARCH
EP_prediction_matrix_24[,13] = c(msgarch_norm_24, rep(0, length(gjrgarch_ged_24)-length(msgarch_norm_24)))
EP_prediction_matrix_24[,14] = c(msgarch_std_24, rep(0, length(gjrgarch_ged_24)-length(msgarch_norm_24)))
EP_prediction_matrix_24[,15] = c(msgarch_ged_24, rep(0, length(gjrgarch_ged_24)-length(msgarch_norm_24)))

# Export prediction matrices
write.table(EP_prediction_matrix_1, "./1_EP/EP_predictions/EP_prediction_matrix_1.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(EP_prediction_matrix_12, "./1_EP/EP_predictions/EP_prediction_matrix_12.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(EP_prediction_matrix_24, "./1_EP/EP_predictions/EP_prediction_matrix_24.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#################### Features: N-step ahead forecasts GARCH-type models ##################

" Here we produce the features needed for the hybrid modeling, where we enrich the 
univariate realized volatility with the garch-type forecasts, for 12-step ahead forecasting
we use the feature of the 12-step ahead forecasts. For 24-step ahead forecasting we, of
course, use the 24-step ahead forecast feature. Therefore we need to have the 24th forecast
at each time point, where we throw the first x observations, equal to the window size, away
as we need these to make the first 12 or 24-step ahead prediction."

window_size = 3000
h = 20000
k = 1
h1 = 12
h2 = 24
ref = 10000

##### h = 1 ####

vec_arch_norm_1 = produce_vector_forecasts_arch("sGARCH", "norm", window_size, log_returns, 1)
vec_arch_std_1 = produce_vector_forecasts_arch("sGARCH", "std", window_size, log_returns, 1)
vec_arch_ged_1 = produce_vector_forecasts_arch("sGARCH", "ged", window_size, log_returns, 1)

vec_garch_norm_1 = produce_vector_forecasts("sGARCH", "norm", window_size, log_returns, 1)
vec_garch_std_1 = produce_vector_forecasts("sGARCH", "std", window_size, log_returns, 1)
vec_garch_ged_1 = produce_vector_forecasts("sGARCH", "ged", window_size, log_returns, 1)

vec_egarch_norm_1 = produce_vector_forecasts("eGARCH", "norm", window_size, log_returns, 1)
vec_egarch_std_1 = produce_vector_forecasts("eGARCH", "std", window_size, log_returns, 1)
vec_egarch_ged_1 = produce_vector_forecasts("eGARCH", "ged", window_size, log_returns, 1)

vec_gjrgarch_norm_1 = produce_vector_forecasts("gjrGARCH", "norm", window_size, log_returns, 1)
vec_gjrgarch_std_1 = produce_vector_forecasts("gjrGARCH", "std", window_size, log_returns, 1)
vec_gjrgarch_ged_1 = produce_vector_forecasts("gjrGARCH", "ged", window_size, log_returns, 1)

vec_msgarch_norm_1 = produce_ms_nstep_features_new(1, "norm", log_returns, h, window_size)
vec_msgarch_std_1 = produce_ms_nstep_features_new(1, "std", log_returns, h, window_size)
vec_msgarch_ged_1 = produce_ms_nstep_features_new(1, "ged", log_returns, h, window_size)


##### h = 12 ####

# ARCH model
vec_arch_norm_12 = produce_vector_forecasts_arch("sGARCH", "norm", window_size, log_returns, h1)
vec_arch_std_12 = produce_vector_forecasts_arch("sGARCH", "std", window_size, log_returns, h1)
vec_arch_ged_12 = produce_vector_forecasts_arch("sGARCH", "ged", window_size, log_returns, h1)

# GARCH model
vec_garch_norm_12 = produce_vector_forecasts("sGARCH", "norm", window_size, log_returns, h1)
vec_garch_std_12 = produce_vector_forecasts("sGARCH", "std", window_size, log_returns, h1)
vec_garch_ged_12 = produce_vector_forecasts("sGARCH", "ged", window_size, log_returns, h1)

# E-GARCH model
vec_egarch_norm_12 = produce_vector_forecasts("eGARCH", "norm", window_size, log_returns, h1)
vec_egarch_std_12 = produce_vector_forecasts("eGARCH", "std", window_size, log_returns, h1)
vec_egarch_ged_12 = produce_vector_forecasts("eGARCH", "ged", window_size, log_returns, h1)

# GJR-GARCH model
vec_gjrgarch_norm_12 = produce_vector_forecasts("gjrGARCH", "norm", window_size, log_returns, h1)
vec_gjrgarch_std_12 = produce_vector_forecasts("gjrGARCH", "std", window_size, log_returns, h1)
vec_gjrgarch_ged_12 = produce_vector_forecasts("gjrGARCH", "ged", window_size, log_returns, h1)

# MS-GARCH model
window_size = 3000
vec_msgarch_norm_12 = produce_ms_nstep_features_new(h1, "norm", log_returns, h, window_size)
vec_msgarch_std_12 = produce_ms_nstep_features_new(h1, "std", log_returns, h, window_size)
vec_msgarch_ged_12 = produce_ms_nstep_features_new(h1, "ged", log_returns, h, window_size)

#### h = 24 ####

# ARCH model
vec_arch_norm_24 = produce_vector_forecasts_arch("sGARCH", "norm", window_size, log_returns, h2)
vec_arch_std_24 = produce_vector_forecasts_arch("sGARCH", "std", window_size, log_returns, h2)
vec_arch_ged_24 = produce_vector_forecasts_arch("sGARCH", "ged", window_size, log_returns, h2)

# GARCH model
vec_garch_norm_24 = produce_vector_forecasts("sGARCH", "norm", window_size, log_returns, h2)
vec_garch_std_24 = produce_vector_forecasts("sGARCH", "std", window_size, log_returns, h2)
vec_garch_ged_24 = produce_vector_forecasts("sGARCH", "ged", window_size, log_returns, h2)

# E-GARCH model
vec_egarch_norm_24 = produce_vector_forecasts("eGARCH", "norm", window_size, log_returns, h2)
vec_egarch_std_24 = produce_vector_forecasts("eGARCH", "std", window_size, log_returns, h2)
vec_egarch_ged_24 = produce_vector_forecasts("eGARCH", "ged", window_size, log_returns, h2)

# GJR-GARCH model
vec_gjrgarch_norm_24 = produce_vector_forecasts("gjrGARCH", "norm", window_size, log_returns, h2)
vec_gjrgarch_std_24 = produce_vector_forecasts("gjrGARCH", "std", window_size, log_returns, h2)
vec_gjrgarch_ged_24 = produce_vector_forecasts("gjrGARCH", "ged", window_size, log_returns, h2)

# MS-GARCH model 
vec_msgarch_norm_24 = produce_ms_nstep_features_new(h2, "norm", log_returns, h, window_size)
vec_msgarch_std_24 = produce_ms_nstep_features_new(h2, "std", log_returns, h, window_size)
vec_msgarch_ged_24 = produce_ms_nstep_features_new(h2, "ged", log_returns, h, window_size)

#RAN UNTIL HERE

#### Export all features: ####
# h = 1
s = 5 # Amount of models
l = 3 # Amount of distributions
EP_feature_matrix_1 = matrix(nrow = length(vec_arch_norm_1), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_feature_matrix_1) = col_names_pred

#ARCH
EP_feature_matrix_1[,1] = vec_arch_norm_1
EP_feature_matrix_1[,2] = vec_arch_std_1
EP_feature_matrix_1[,3] = vec_arch_ged_1
#GARCH
EP_feature_matrix_1[,4] = vec_garch_norm_1
EP_feature_matrix_1[,5] = vec_garch_std_1
EP_feature_matrix_1[,6] = vec_garch_ged_1
#E-GARCH
EP_feature_matrix_1[,7] = vec_egarch_norm_1
EP_feature_matrix_1[,8] = vec_egarch_std_1
EP_feature_matrix_1[,9] = vec_egarch_ged_1
#GJR-GARCH
EP_feature_matrix_1[,10] = vec_gjrgarch_norm_1
EP_feature_matrix_1[,11] = vec_gjrgarch_std_1
EP_feature_matrix_1[,12] = vec_gjrgarch_ged_1
#MS-GARCH
EP_feature_matrix_1[,13] = c(0,vec_msgarch_norm_1) 
EP_feature_matrix_1[,14] = c(0,vec_msgarch_std_1) 
EP_feature_matrix_1[,15] = c(0,vec_msgarch_ged_1) 

# h = 12
s = 5 # Amount of models
l = 3 # Amount of distributions
EP_feature_matrix_12 = matrix(nrow = length(vec_arch_norm_12), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_feature_matrix_12) = col_names_pred

#ARCH
EP_feature_matrix_12[,1] = vec_arch_norm_12
EP_feature_matrix_12[,2] = vec_arch_std_12
EP_feature_matrix_12[,3] = vec_arch_ged_12
#GARCH
EP_feature_matrix_12[,4] = vec_garch_norm_12
EP_feature_matrix_12[,5] = vec_garch_std_12
EP_feature_matrix_12[,6] = vec_garch_ged_12
#E-GARCH
EP_feature_matrix_12[,7] = vec_egarch_norm_12
EP_feature_matrix_12[,8] = vec_egarch_std_12
EP_feature_matrix_12[,9] = vec_egarch_ged_12
#GJR-GARCH
EP_feature_matrix_12[,10] = vec_gjrgarch_norm_12
EP_feature_matrix_12[,11] = vec_gjrgarch_std_12
EP_feature_matrix_12[,12] = vec_gjrgarch_ged_12
#MS-GARCH
EP_feature_matrix_12[,13] = c(rep(0, 12), vec_msgarch_norm_12)  
EP_feature_matrix_12[,14] = c(rep(0, 12), vec_msgarch_std_12)  
EP_feature_matrix_12[,15] = c(rep(0, 12), vec_msgarch_ged_12)   

# h = 24
s = 5 # Amount of models
l = 3 # Amount of distributions
EP_feature_matrix_24 = matrix(nrow = length(vec_arch_norm_24), ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                   "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                   "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(EP_feature_matrix_24) = col_names_pred

#ARCH
EP_feature_matrix_24[,1] = vec_arch_norm_24
EP_feature_matrix_24[,2] = vec_arch_std_24
EP_feature_matrix_24[,3] = vec_arch_ged_24
#GARCH
EP_feature_matrix_24[,4] = vec_garch_norm_24
EP_feature_matrix_24[,5] = vec_garch_std_24
EP_feature_matrix_24[,6] = vec_garch_ged_24
#E-GARCH
EP_feature_matrix_24[,7] = vec_egarch_norm_24
EP_feature_matrix_24[,8] = vec_egarch_std_24
EP_feature_matrix_24[,9] = vec_egarch_ged_24
#GJR-GARCH
EP_feature_matrix_24[,10] = vec_gjrgarch_norm_24
EP_feature_matrix_24[,11] = vec_gjrgarch_std_24
EP_feature_matrix_24[,12] = vec_gjrgarch_ged_24
#MS-GARCH
EP_feature_matrix_24[,13] = c(rep(0, 24), vec_msgarch_norm_24) 
EP_feature_matrix_24[,14] = c(rep(0, 24), vec_msgarch_std_24) 
EP_feature_matrix_24[,15] = c(rep(0, 24), vec_msgarch_ged_24)

plot(vec_msgarch_std_24, type = 'l')

# Export tables!
write.table(EP_feature_matrix_1, "./1_EP/EP_features/Portugal/EP_feature_matrix_portugal_1.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(EP_feature_matrix_12, "./1_EP/EP_features/Portugal/EP_feature_matrix_portugal_12.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(EP_feature_matrix_24, "./1_EP/EP_features/Portugal/EP_feature_matrix_portugal_24.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
