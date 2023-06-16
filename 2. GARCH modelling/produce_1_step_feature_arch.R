produce_1_step_feature_arch = function(model, distr, log_returns, log_returns_train, h, ref){
  
  #First: 1-step ahead forecasts (out-of-sample)
  error_mod_distr = forecast_eval_arch(log_returns, model, distr, h=h, refit_every = ref)
  
  #Secon: 1-step ahead forecasts (in-sample) 
  spec <- ugarchspec(variance.model = list(model = model, garchOrder = c(3, )),
                     mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                     distribution.model = distr)
  fit <- ugarchfit(spec, data = log_returns_train)
  volatility <- as.numeric(sigma(fit))
  export_gjr_std = c(volatility, error_e_std[,1])
}