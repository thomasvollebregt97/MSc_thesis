produce_ms_nstep_forecasts = function(n_ahead, distr, log_returns, h){

  size = round(h / n_ahead) #Maximum: round(h / n_ahead), i.e. round(20.000 / 12) 
  buf = 50000
  ms_h_step_forecast = numeric()
  
  pb <- txtProgressBar(min = 0, max = size, style = 3)
  
  for (i in 1:(size-1)){
    MSmodel = CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH")),
                         distribution.spec = list(distribution = c(distr, distr)),
                         switch.spec = list(do.mix = FALSE))
    MSmodel_est<-FitML(spec = MSmodel, data = log_returns[(buf+(i*n_ahead)):(i*n_ahead+(length(log_returns)-h))])
    MSforecast <- predict(MSmodel_est, nahead = n_ahead)
    ms_h_step_forecast <- c(ms_h_step_forecast, MSforecast$vol)
    setTxtProgressBar(pb, i)
  }
  return(ms_h_step_forecast)
}