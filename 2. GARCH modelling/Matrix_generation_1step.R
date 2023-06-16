# Matrix generation

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

log_returns = read.csv("log_returns_spain.csv", header = TRUE)
log_returns = log_returns$Price..EUR.MWhe.
h = 20000
refit_every = 10000

matrices = generate_matrix_results(log_returns, h, refit_every)

matrix_predictions = matrices$matrix_1
matrix_filtered = matrices$matrix_2

# Export tables
write.table(matrix_predictions, "./1_Predicted_garch/matrix_predictions.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(matrix_filtered, "./2_Filtered_garch/matrix_filtereds.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
