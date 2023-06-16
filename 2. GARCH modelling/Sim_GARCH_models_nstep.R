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
source('generate_filter_arch.R')
source('generate_filter.R')

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

" This file turn 1 sequence of simulated log-returns into 3 matrices, 1 for each forecasting horizon.
REPEAT BLOCK ARCH FOR EACH SEQUENCE, WHEN VERIFIED THAT IT WORKS!"


###################################### Importing the simulated sequences ######################################
sim_arch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_arch_draws.csv", header = TRUE)$Col1
sim_garch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_garch_draws.csv", header = TRUE)$Col1
sim_egarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_egarch_draws.csv", header = TRUE)$Col1
sim_gjrgarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_gjrgarch_draws.csv", header = TRUE)$Col1
sim_msgarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_msgarch_draws.csv", header = TRUE)$Col1


# Define parameters
h = 20000 #(~30%) Forecast horizon, choose a multiple of the refit.every variable
ref = 10000
window_size = 2000
h1 = 12
h2 = 24


####################################################   ARCH: Std   ##############################################
sim_arch_train = sim_arch_draw[1:(length(sim_arch_draw)-h)]
sim_arch_test = sim_arch_draw[(length(sim_arch_draw)-h+1):(length(sim_arch_draw))]

##### Features: ####
window_size = 3000
h0 = 1
h1 = 12
h2 = 24

# 1-step ahead features
feat_arch_1_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_arch_draw, h0)
feat_arch_1_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_arch_draw, h0)
feat_arch_1_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_arch_draw, h0)
feat_arch_1_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_arch_draw, h0)
feat_arch_1_msgarch =  produce_forecast_and_filter_ms("std", sim_arch_train, sim_arch_test)
diff = length(feat_arch_1_msgarch) - length(feat_arch_1_msgarch)
feat_arch_1_msgarch = feat_arch_1_msgarch[(diff+1):length(feat_arch_1_msgarch)]

# 12-step ahead features
feat_arch_12_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_arch_draw, h1)
feat_arch_12_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_arch_draw, h1)
feat_arch_12_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_arch_draw, h1)
feat_arch_12_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_arch_draw, h1)
feat_arch_12_msgarch =  rep(0,length(feat_arch_12_gjrgarch))

# 24-step ahead features
feat_arch_24_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_arch_draw, h2)
feat_arch_24_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_arch_draw, h2)
feat_arch_24_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_arch_draw, h2)
feat_arch_24_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_arch_draw, h2)
feat_arch_24_msgarch =  rep(0,length(feat_arch_24_gjrgarch))

# Export features based on arch simulated sequence
write.table(feat_arch_1_arch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_1_garch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_1_egarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_1_gjrgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_1_msgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_arch_12_arch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_12_garch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_12_egarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_12_gjrgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_12_msgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_arch_24_arch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_24_garch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_24_egarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_24_gjrgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_arch_24_msgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  GARCH: Std   ###################################################
sim_garch_train = sim_garch_draw[1:(length(sim_garch_draw)-h)]
sim_garch_test = sim_garch_draw[(length(sim_garch_draw)-h+1):(length(sim_garch_draw))]

##### Features: ####
window_size = 3000
h0 = 1
h1 = 12
h2 = 24

# 1-step ahead features
feat_garch_1_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_garch_draw, h0)
feat_garch_1_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_garch_draw, h0)
feat_garch_1_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_garch_draw, h0)
feat_garch_1_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_garch_draw, h0)
feat_garch_1_msgarch =  produce_forecast_and_filter_ms("std", sim_garch_train, sim_garch_test)
diff = length(feat_garch_1_msgarch) - length(feat_garch_1_msgarch)
feat_garch_1_msgarch = feat_garch_1_msgarch[(diff+1):length(feat_garch_1_msgarch)]

# 12-step ahead features
feat_garch_12_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_garch_draw, h1)
feat_garch_12_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_garch_draw, h1)
feat_garch_12_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_garch_draw, h1)
feat_garch_12_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_garch_draw, h1)
feat_garch_12_msgarch =  rep(0, length(feat_garch_12_gjrgarch))

# 24-step ahead features
feat_garch_24_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_garch_draw, h2)
feat_garch_24_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_garch_draw, h2)
feat_garch_24_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_garch_draw, h2)
feat_garch_24_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_garch_draw, h2)
feat_garch_24_msgarch =  rep(0, length(feat_garch_24_gjrgarch))

# Export features based on arch simulated sequence
write.table(feat_garch_1_arch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_1_garch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_1_egarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_1_gjrgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_1_msgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_garch_12_arch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_12_garch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_12_egarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_12_gjrgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_12_msgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_garch_24_arch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_24_garch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_24_egarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_24_gjrgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_garch_24_msgarch, "./2_Simulated_sequences/Features_simulations/GARCH sequence/GARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  E-GARCH: Std  ###################################################
sim_egarch_train = sim_egarch_draw[1:(length(sim_egarch_draw)-h)]
sim_egarch_test = sim_egarch_draw[(length(sim_egarch_draw)-h+1):(length(sim_egarch_draw))]

##### Features: ####
window_size = 3000
h0 = 1
h1 = 12
h2 = 24

# 1-step ahead features
feat_egarch_1_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_egarch_draw, h0)
feat_egarch_1_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_egarch_draw, h0)
feat_egarch_1_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_egarch_draw, h0)
feat_egarch_1_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_egarch_draw, h0)
feat_egarch_1_msgarch =  produce_forecast_and_filter_ms("std", sim_egarch_train, sim_egarch_test)
diff = length(feat_egarch_1_msgarch) - length(feat_egarch_1_msgarch)
feat_egarch_1_msgarch = feat_egarch_1_msgarch[(diff+1):length(feat_egarch_1_msgarch)]

# 12-step ahead features
feat_egarch_12_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_egarch_draw, h1)
feat_egarch_12_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_egarch_draw, h1)
feat_egarch_12_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_egarch_draw, h1)
feat_egarch_12_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_egarch_draw, h1)
feat_egarch_12_msgarch =  rep(0, length(feat_egarch_12_gjrgarch))

# 24-step ahead features
feat_egarch_24_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_egarch_draw, h2)
feat_egarch_24_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_egarch_draw, h2)
feat_egarch_24_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_egarch_draw, h2)
feat_egarch_24_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_egarch_draw, h2)
feat_egarch_24_msgarch =  rep(0, length(feat_egarch_24_gjrgarch))

# Export features based on arch simulated sequence
write.table(feat_egarch_1_arch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_1_garch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_1_egarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_1_gjrgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_1_msgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_egarch_12_arch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_12_garch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_12_egarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_12_gjrgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_12_msgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_egarch_24_arch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_24_garch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_24_egarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_24_gjrgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_egarch_24_msgarch, "./2_Simulated_sequences/Features_simulations/E-GARCH sequence/EGARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  GJR-GARCH: Std  ###################################################
sim_gjrgarch_train = sim_gjrgarch_draw[1:(length(sim_gjrgarch_draw)-h)]
sim_gjrgarch_test = sim_gjrgarch_draw[(length(sim_gjrgarch_draw)-h+1):(length(sim_gjrgarch_draw))]

##### Features: ####
window_size = 3000
h0 = 1
h1 = 12
h2 = 24

# 1-step ahead features
feat_gjrgarch_1_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_gjrgarch_draw, h0)
feat_gjrgarch_1_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_gjrgarch_draw, h0)
feat_gjrgarch_1_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_gjrgarch_draw, h0)
feat_gjrgarch_1_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_gjrgarch_draw, h0)
feat_gjrgarch_1_msgarch =  produce_forecast_and_filter_ms("std", sim_gjrgarch_train, sim_gjrgarch_test)
diff = length(feat_gjrgarch_1_msgarch) - length(feat_gjrgarch_1_msgarch)
feat_gjrgarch_1_msgarch = feat_gjrgarch_1_msgarch[(diff+1):length(feat_gjrgarch_1_msgarch)]

# 12-step ahead features
feat_gjrgarch_12_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_gjrgarch_draw, h1)
feat_gjrgarch_12_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_gjrgarch_draw, h1)
feat_gjrgarch_12_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_gjrgarch_draw, h1)
feat_gjrgarch_12_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_gjrgarch_draw, h1)
feat_gjrgarch_12_msgarch =  rep(0, length(feat_gjrgarch_12_gjrgarch))

# 24-step ahead features
feat_gjrgarch_24_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_gjrgarch_draw, h2)
feat_gjrgarch_24_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_gjrgarch_draw, h2)
feat_gjrgarch_24_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_gjrgarch_draw, h2)
feat_gjrgarch_24_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_gjrgarch_draw, h2)
feat_gjrgarch_24_msgarch =  rep(0, length(feat_gjrgarch_24_gjrgarch))

# Export features based on arch simulated sequence
write.table(feat_gjrgarch_1_arch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_1_garch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_1_egarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_1_gjrgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_1_msgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_gjrgarch_12_arch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_12_garch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_12_egarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_12_gjrgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_12_msgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export features based on arch simulated sequence
write.table(feat_gjrgarch_24_arch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_24_garch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_24_egarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_24_gjrgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_gjrgarch_24_msgarch, "./2_Simulated_sequences/Features_simulations/GJR-GARCH sequence/GJRGARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  MS-GARCH: Std  ###################################################

sim_msgarch_train = sim_msgarch_draw[1:(length(sim_msgarch_draw)-h)]
sim_msgarch_test = sim_msgarch_draw[(length(sim_msgarch_draw)-h+1):(length(sim_msgarch_draw))]

##### Features: ####
window_size = 3000
h0 = 1
h1 = 12
h2 = 24

# 1-step ahead features
feat_msgarch_1_arch = produce_vector_forecasts_arch("sGARCH", "std", window_size, sim_msgarch_draw, h0)
feat_msgarch_1_garch = produce_vector_forecasts("sGARCH", "std", window_size, sim_msgarch_draw, h0)
feat_msgarch_1_egarch = produce_vector_forecasts("eGARCH", "std", window_size, sim_msgarch_draw, h0)
feat_msgarch_1_gjrgarch = produce_vector_forecasts("gjrGARCH", "std", window_size, sim_msgarch_draw, h0)
feat_msgarch_1_msgarch =  produce_forecast_and_filter_ms("std", sim_msgarch_train, sim_msgarch_test)
diff = length(feat_msgarch_1_msgarch) - length(feat_msgarch_1_msgarch)
feat_msgarch_1_msgarch = feat_msgarch_1_msgarch[(diff+1):length(feat_msgarch_1_msgarch)]

# Export features based on arch simulated sequence
write.table(feat_msgarch_1_arch, "./2_Simulated_sequences/Features_simulations/MS-GARCH sequence/MSGARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_msgarch_1_garch, "./2_Simulated_sequences/Features_simulations/MS-GARCH sequence/MSGARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_msgarch_1_egarch, "./2_Simulated_sequences/Features_simulations/MS-GARCH sequence/MSGARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_msgarch_1_gjrgarch, "./2_Simulated_sequences/Features_simulations/MS-GARCH sequence/MSGARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(feat_msgarch_1_msgarch, "./2_Simulated_sequences/Features_simulations/MS-GARCH sequence/MSGARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)




############################################### Predictions: (redundant ?) ################################################

# 1-step ahead prediction
pred_arch_1_arch = feat_arch_1_arch[(length(feat_arch_1_arch)-h):length(feat_arch_1_arch)]
pred_arch_1_garch = feat_arch_1_garch[(length(feat_arch_1_garch)-h):length(feat_arch_1_garch)]
pred_arch_1_egarch = feat_arch_1_egarch[(length(feat_arch_1_egarch)-h):length(feat_arch_1_egarch)]
pred_arch_1_gjrgarch = feat_arch_1_gjrgarch[(length(feat_arch_1_gjrgarch)-h):length(feat_arch_1_gjrgarch)]
pred_arch_1_msgarch = feat_arch_1_msgarch[(length(feat_arch_1_msgarch)-h):length(feat_arch_1_msgarch)]

# 12-step ahead prediction
pred_arch_12_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_arch_draw, h1)
pred_arch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h1)
pred_arch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h1)
pred_arch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h1)
pred_arch_12_garch = produce_ms_nstep_forecasts(h1, "std", sim_arch_draw, h)

# 24-step ahead prediction
pred_arch_24_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_arch_draw, h2)
pred_arch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h2)
pred_arch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h2)
pred_arch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_arch_draw, h2)
pred_arch_24_garch = produce_ms_nstep_forecasts(h2, "std", sim_arch_draw, h)

# Export predictions based on arch simulated sequence
write.table(pred_arch_1_arch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_1_garch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_1_egarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_1_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_1_msgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_arch_12_arch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_12_garch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_12_egarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_12_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_12_msgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_arch_24_arch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_24_garch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_24_egarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_24_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_arch_24_msgarch, "./2_Simulated_sequences/Predictions_simulations/ARCH sequence/ARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)



##########################################  GARCH ################################### 

# 1-step ahead prediction
pred_garch_1_arch = feat_garch_1_arch[(length(feat_garch_1_arch)-h):length(feat_garch_1_arch)]
pred_garch_1_garch = feat_garch_1_garch[(length(feat_garch_1_garch)-h):length(feat_garch_1_garch)]
pred_garch_1_egarch = feat_garch_1_egarch[(length(feat_garch_1_egarch)-h):length(feat_garch_1_egarch)]
pred_garch_1_gjrgarch = feat_garch_1_gjrgarch[(length(feat_garch_1_gjrgarch)-h):length(feat_garch_1_gjrgarch)]
pred_garch_1_msgarch = feat_garch_1_msgarch[(length(feat_garch_1_msgarch)-h):length(feat_garch_1_msgarch)]

# 12-step ahead prediction
pred_garch_12_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_garch_draw, h1)
pred_garch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h1)
pred_garch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h1)
pred_garch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h1)
pred_garch_12_garch = produce_ms_nstep_forecasts(h1, "std", sim_garch_draw, h)

# 24-step ahead prediction
pred_garch_24_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_garch_draw, h2)
pred_garch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h2)
pred_garch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h2)
pred_garch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_garch_draw, h2)
pred_garch_24_garch = produce_ms_nstep_forecasts(h2, "std", sim_garch_draw, h)

# Export predictions based on arch simulated sequence
write.table(pred_garch_1_arch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_garch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_egarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_msgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_garch_12_arch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_garch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_egarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_msgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_garch_24_arch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_garch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_egarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_msgarch, "./2_Simulated_sequences/Predictions_simulations/GARCH sequence/GARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)

########################################   E-GARCH: #######################################       

# 1-step ahead prediction
pred_egarch_1_arch = feat_egarch_1_arch[(length(feat_egarch_1_arch)-h):length(feat_egarch_1_arch)]
pred_egarch_1_garch = feat_egarch_1_garch[(length(feat_egarch_1_garch)-h):length(feat_egarch_1_garch)]
pred_egarch_1_egarch = feat_egarch_1_egarch[(length(feat_egarch_1_egarch)-h):length(feat_egarch_1_egarch)]
pred_egarch_1_gjrgarch = feat_egarch_1_gjrgarch[(length(feat_egarch_1_gjrgarch)-h):length(feat_egarch_1_gjrgarch)]
pred_egarch_1_msgarch = feat_egarch_1_msgarch[(length(feat_egarch_1_msgarch)-h):length(feat_egarch_1_msgarch)]

# 12-step ahead prediction
pred_egarch_12_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_egarch_draw, h1)
pred_egarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h1)
pred_egarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h1)
pred_egarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h1)
pred_egarch_12_garch = produce_ms_nstep_forecasts(h1, "std", sim_egarch_draw, h)

# 24-step ahead prediction
pred_egarch_24_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_egarch_draw, h2)
pred_egarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h2)
pred_egarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h2)
pred_egarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_egarch_draw, h2)
pred_egarch_24_garch = produce_ms_nstep_forecasts(h2, "std", sim_egarch_draw, h)

# Export predictions based on arch simulated sequence
write.table(pred_garch_1_arch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_garch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_egarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_1_msgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_garch_12_arch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_garch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_egarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_12_msgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_garch_24_arch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_garch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_egarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_garch_24_msgarch, "./2_Simulated_sequences/Predictions_simulations/E-GARCH sequence/EGARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


######################################## GJR-GARCH: ############################################

# 1-step ahead prediction
pred_gjrgarch_1_arch = feat_gjrgarch_1_arch[(length(feat_gjrgarch_1_arch)-h):length(feat_gjrgarch_1_arch)]
pred_gjrgarch_1_garch = feat_gjrgarch_1_garch[(length(feat_gjrgarch_1_garch)-h):length(feat_gjrgarch_1_garch)]
pred_gjrgarch_1_egarch = feat_gjrgarch_1_egarch[(length(feat_gjrgarch_1_egarch)-h):length(feat_gjrgarch_1_egarch)]
pred_gjrgarch_1_gjrgarch = feat_gjrgarch_1_gjrgarch[(length(feat_gjrgarch_1_gjrgarch)-h):length(feat_gjrgarch_1_gjrgarch)]
pred_gjrgarch_1_msgarch = feat_gjrgarch_1_msgarch[(length(feat_gjrgarch_1_msgarch)-h):length(feat_gjrgarch_1_msgarch)]

# 12-step ahead prediction
pred_gjrgarch_12_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_gjrgarch_draw, h1)
pred_gjrgarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h1)
pred_gjrgarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h1)
pred_gjrgarch_12_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h1)
pred_gjrgarch_12_garch = produce_ms_nstep_forecasts(h1, "std", sim_gjrgarch_draw, h)

# 24-step ahead prediction
pred_gjrgarch_24_arch = generate_n_step_forecasts_arch("sGARCH", "std", h, sim_gjrgarch_draw, h2)
pred_gjrgarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h2)
pred_gjrgarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h2)
pred_gjrgarch_24_garch = generate_n_step_forecasts("sGARCH", "std", h, sim_gjrgarch_draw, h2)
pred_gjrgarch_24_garch = produce_ms_nstep_forecasts(h2, "std", sim_gjrgarch_draw, h)

# Export predictions based on arch simulated sequence
write.table(pred_gjrgarch_1_arch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_1_garch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_1_egarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_1_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_1_msgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_gjrgarch_12_arch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_12_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_12_garch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_12_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_12_egarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_12_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_12_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_12_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_12_msgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_12_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
# Export predictions based on arch simulated sequence
write.table(pred_gjrgarch_24_arch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_24_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_24_garch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_24_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_24_egarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_24_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_24_gjrgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_24_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(pred_gjrgarch_24_msgarch, "./2_Simulated_sequences/Predictions_simulations/GJR-GARCH sequence/GJRGARCH_24_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#################################################### MS-GARCH (TBC) ####################################################

