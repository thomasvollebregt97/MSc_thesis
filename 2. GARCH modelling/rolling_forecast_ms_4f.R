rolling_forecast_ms_4f = function(distr, train_1, test_1, train_2, test_2, train_3, test_3, train_4, test_4, rv_test){
  forecast_1 = forecast_eval_ms_plus(distr, train_1, test_1) #for generating 1 extra filtered observation
  forecast_2 = forecast_eval_ms(distr, train_2, test_2)
  forecast_3 = forecast_eval_ms(distr, train_3, test_3)
  forecast_4 = forecast_eval_ms(distr, train_4, test_4)
  
  rolling_forecast_ms = c(forecast_1, forecast_2,forecast_3, forecast_4) #Refitted every 5000 observations
  rolling_forecast_ms = head(rolling_forecast_ms, -1) #Export this rolling forecast
  
  rv_test = sqrt(rv_test)
  
  # Calculating MSE
  MSE = (1/length(rv_test))*sum((rolling_forecast_ms - rv_test)^2)
  # Calculating MAE
  MAE = (1/length(rv_test))*sum(abs(rolling_forecast_ms - rv_test))
  # Allow for other error metrics
  error = numeric(h)
  error[1] = MSE
  error[2] = MAE
  matrix_s = cbind(rolling_forecast_ms, rv_test, error)
  return(matrix_s)
  
  return(rolling_forecast_ms)
  }