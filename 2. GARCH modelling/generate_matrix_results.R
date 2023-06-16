generate_matrix_results =  function(log_returns, h, refit_every){

  #Splitting dataset set
  h = h #(~30%) Forecast horizon, choose a multiple of the refit.every variable
  ref = refit_every
  log_returns_train = log_returns[1:(length(log_returns)-h)]
  log_returns_test = log_returns[(length(log_returns)-h+1):(length(log_returns))]
  
  
  #### Create & evaluate forecasts (1-step ahead rolling forecasts) ####
  
  #sARCH (5 lags chosen...)
  error_a_norm = forecast_eval_arch(log_returns, "sGARCH", "norm", h = h, refit_every = ref)
  error_a_std = forecast_eval_arch(log_returns, "sGARCH", "std", h = h, refit_every = ref)
  error_a_ged = forecast_eval_arch(log_returns, "sGARCH", "ged", h = h, refit_every = ref) # Use: solver = "gosolnp"
  
  #sGARCH
  error_s_norm = forecast_eval(log_returns, "sGARCH", "norm", h = h, refit_every = ref)
  error_s_std = forecast_eval(log_returns, "sGARCH", "std", h = h, refit_every = ref)
  error_s_ged = forecast_eval(log_returns, "sGARCH", "ged", h = h, refit_every = ref)
  
  #gjrGARCH
  error_g_norm = forecast_eval(log_returns, "gjrGARCH", "norm", h = h, refit_every = ref)
  error_g_std = forecast_eval(log_returns, "gjrGARCH", "std", h = h, refit_every = ref)
  error_g_ged = forecast_eval(log_returns, "gjrGARCH", "ged", h = h, refit_every = ref)
  
  #### eGARCH - Does not seem to work: explanation in report why? ####
  error_e_norm = forecast_eval(log_returns, "eGARCH", "norm", h=h, refit_every = ref)
  error_e_std = forecast_eval(log_returns, "eGARCH", "std", h=h,  refit_every = ref)
  error_e_ged = forecast_eval(log_returns, "eGARCH", "ged", h=h, refit_every = ref)
  
  
  ### Training set for rolling forecast ms-garch
  train_1 = log_returns_train
  train_2 = c(log_returns_train[(ref+1):length(log_returns_train)],log_returns_test[1:ref])
  train_2_std = c(log_returns_train[(ref+1):length(log_returns_train)],log_returns_test[1:3500],log_returns_test[3750:5000],log_returns_test[(ref/2):ref])
  #train_3 = c(log_returns_train[(2*ref+1):length(log_returns_train)], log_returns_test[2:(2*ref)]) #Doesn't converge with 1:2*ref...
  #train_4 = c(log_returns_train[(3*ref+1):length(log_returns_train)], log_returns_test[10:(3*ref)])
  
  test_1 = log_returns_test[1:ref]
  test_2 = log_returns_test[(ref+1):(2*ref)]
  #test_3 = log_returns_test[(2*ref+1):(3*ref)]
  #test_4 = log_returns_test[(3*ref+1):(4*ref)]
  
  rv_test = error_a_norm[,2]
  
  #MSGARCH (based on 20.000 test set and every 5000 a refit)
  error_ms_norm = rolling_forecast_ms("norm", train_1, test_1, train_2, test_2, rv_test)
  error_ms_std = rolling_forecast_ms("std", train_1, test_1, train_2_std, test_2, rv_test)
  error_ms_ged = rolling_forecast_ms("ged", train_1, test_1, train_2, test_2, rv_test)
  
  
  ####  ####
  # Export forecasts
  s = 5 # Amount of models
  l = 3 # Amount of distributions
  matrix_predictions = matrix(nrow = h, ncol = s*l)
  
  col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                     "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                     "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
  colnames(matrix_predictions) = col_names_pred
  
  #ARCH
  matrix_predictions[,1] = error_a_norm[,1]
  matrix_predictions[,2] = error_a_std[,1]
  matrix_predictions[,3] = error_a_ged[,1]
  #GARCH
  matrix_predictions[,4] = error_s_norm[,1]
  matrix_predictions[,5] = error_s_std[,1]
  matrix_predictions[,6] = error_s_ged[,1]
  #E-GARCH
  matrix_predictions[,7] = error_e_norm[,1]
  matrix_predictions[,8] = error_e_std[,1]
  matrix_predictions[,9] = error_e_ged[,1]
  #GJR-GARCH
  matrix_predictions[,10] = error_g_norm[,1]
  matrix_predictions[,11] = error_g_std[,1]
  matrix_predictions[,12] = error_g_ged[,1]
  #MS-GARCH
  matrix_predictions[,13] = error_ms_norm[,1]
  matrix_predictions[,14] = error_ms_std[,1]
  matrix_predictions[,15] = error_ms_ged[,1]
  
  #### Export filtered sequence ####
  
  s = 2 # Amount of models
  l = 2 # Amount of distributions
  matrix_filtered = matrix(nrow = length(log_returns), ncol = s*l)
  
  col_names_filt = c("E-GARCH-Std","E-GARCH-GED","MS-GARCH-Std","MS-GARCH-GED")
  colnames(matrix_filtered) = col_names_filt
  
  # MS-GARCH std/ged
  filt_ms_std = produce_forecast_and_filter_ms("std", log_returns_train, log_returns_test)
  filt_ms_ged = produce_forecast_and_filter_ms("ged", log_returns_train, log_returns_test)
  
  #E-GARCH: STD
  spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = "std")
  fit <- ugarchfit(spec, data = log_returns_train)
  volatility <- as.numeric(sigma(fit))
  export_gjr_std = c(volatility, error_e_std[,1])
  
  #E-GARCH: GED
  spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = "ged")
  fit <- ugarchfit(spec, data = log_returns_train)
  volatility <- as.numeric(sigma(fit))
  export_gjr_ged = c(volatility, error_e_ged[,1])
  
  matrix_filtered[,1] = filt_ms_std
  matrix_filtered[,2] = filt_ms_ged
  matrix_filtered[,3] = export_gjr_std
  matrix_filtered[,4] = export_gjr_ged
  
  output = list(matrix_1 = matrix_predictions, matrix_2 = matrix_filtered)
  return(output)
}