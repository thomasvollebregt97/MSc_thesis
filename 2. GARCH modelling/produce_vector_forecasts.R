produce_vector_forecasts = function(model, distr, window_size, log_returns, n_ahead){
  
  #### Generate n step ahead forecasts for entire set
  
  # Fit GARCH-type model
  spec = ugarchspec(variance.model = list(model = model, garchOrder = c(1, 1)),
                    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                    distribution.model = distr)
  
  window_size = window_size
  out_sample = length(log_returns) - window_size
  
  garch_fit = ugarchfit(spec = spec, data = log_returns, out.sample = out_sample)
  
  # Generate n-step forecasts
  
  n_ahead = n_ahead
  nsteproll_forecast = ugarchforecast(garch_fit, n.ahead = n_ahead, n.roll = out_sample)
  n_step_forecast <- nsteproll_forecast@forecast$sigmaFor[n_ahead,]
}