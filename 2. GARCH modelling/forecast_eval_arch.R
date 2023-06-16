forecast_eval_arch = function(data, model, distr, h, refit_every){
  
  log_returns = data
  
  # Calculating realized volatility
  horizon = 100
  rv = vector(mode = "numeric", length = length(log_returns) - (horizon-1))
  
  #Shifting the window 1-step at a time
  for (i in 1:(length(rv))) {
    rv[i] = var(log_returns[(i):((horizon-1)+i)]) #Realized volatility is set equal to the variance of the subset of log-returns
  }
  
  #Drop first 99 observations of log returns
  log_returns = log_returns[100:length(log_returns)] #Given each log return, the associated realized volatility is calculated
  
  #Train set
  h = h #(20000 =~ 30%) Forecast horizon, choose a multiple of the refit.every variable
  log_returns_train = log_returns[1:(length(log_returns)-h)]
  
  #Test set
  rv_test = rv[(length(log_returns)-(h-1)):length(log_returns)]
  rv_test = sqrt(rv_test)
  
  #### Create forecasts ####
  
  ###### Working rolling forecast Specify GARCH model
  spec <- ugarchspec(variance.model = list(model = model, garchOrder = c(1, 0)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = distr)
  
  n_start = length(log_returns_train)
  
  roll = ugarchroll(spec, log_returns, n.ahead = 1, 
                    n.start = n_start, refit.every = refit_every, solver = "gosolnp") #Try different solver for ARCH - GED
  #Window size seems equal to starting point
  
  # Roll GARCH model forward in time to produce forecast & extract forecasted volatility
  forecast <- roll@forecast$density$Sigma
  #### Evaluate forecasts (1-step ahead rolling forecasts) ####
  
  # Calculating MSE
  MSE = (1/length(rv_test))*sum((forecast - rv_test)^2)
  # Calculating MAE
  MAE = (1/length(rv_test))*sum(abs(forecast - rv_test))
  # Allow for other error metrics
  error = numeric(h)
  error[1] = MSE
  error[2] = MAE
  matrix_s = cbind(forecast, rv_test, error)
  return(matrix_s)
}