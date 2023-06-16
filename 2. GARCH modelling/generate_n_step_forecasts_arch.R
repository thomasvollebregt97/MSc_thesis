generate_n_step_forecasts_arch = function(model, distr, h, log_returns, n_ahead){
  #### Generate n step ahead forecasts
  
  # Fit GARCH-type model
  spec = ugarchspec(variance.model = list(model = model, garchOrder = c(3, 0)),
                    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                    distribution.model = distr)
  garch_fit = ugarchfit(spec = spec, data = log_returns, out.sample = h)
  
  # Generate n-step forecasts
  n_ahead = n_ahead
  nsteproll_forecast = ugarchforecast(garch_fit, n.ahead = n_ahead, n.roll = h)
  
  n_step_forecast <- numeric()
  
  for (i in seq(1, h, by = n_ahead) ){
    n_step_forecast <- c(n_step_forecast, nsteproll_forecast@forecast$sigmaFor[, i])
  }
  
  return(n_step_forecast)
}