generate_filter = function(log_returns, log_returns_train, h, ref, model){
  
  error_e_ged = forecast_eval(log_returns, model, "std", h=h, refit_every = ref)
  
  spec <- ugarchspec(variance.model = list(model = model, garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = "std")
  
  fit <- ugarchfit(spec, data = log_returns_train, solver = 'BFGS') #Hybrid for most optimizations
  volatility <- as.numeric(sigma(fit))
  
  filtered_arch = c(volatility, error_e_ged[,1])
  return(filtered_arch)
}