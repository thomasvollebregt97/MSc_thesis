generate_filter_arch = function(log_returns, log_returns_train, h, ref){
  
  error_e_ged = forecast_eval(log_returns, "sGARCH", "std", h=h, refit_every = ref)
  
  spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(3, 0)),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = "std")
  
  fit <- ugarchfit(spec, data = log_returns_train, solver = 'hybrid', solver.control = list(tol = 1e-12))
  volatility <- as.numeric(sigma(fit))
  
  filtered_arch = c(volatility, error_e_ged[,1])
  return(filtered_arch)
}