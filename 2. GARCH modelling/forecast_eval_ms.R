forecast_eval_ms = function(distr, log_returns_train, log_returns_test){
  
  model = "sGARCH"
  distribution = distr
  regimes = 2
  
  MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                       distribution.spec = list(distribution = c(distribution)), #'norm', 'std', 'ged'
                       switch.spec = list(do.mix = FALSE, K = regimes))
  
  ms_fit = FitML(MSGARCH, data = log_returns_train)
  
  filt_var = Volatility(ms_fit, newdata = log_returns_test)
  
  filt_var = filt_var[(length(log_returns_train)+1):(length(log_returns_train)+length(log_returns_test))]
  
  return(filt_var)
}