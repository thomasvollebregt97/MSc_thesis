produce_ms_nstep_features_new = function(n_ahead, distr, log_returns, h, window_size){
  
  size = (length(log_returns)-window_size)/n_ahead #Maximum: round(h / n_ahead), i.e. round(20.000 / 12) 
  ms_h_step_forecast = numeric()
  pb <- txtProgressBar(min = 0, max = size, style = 3)
  
  MSmodel = CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH")),
                       distribution.spec = list(distribution = c(distr, distr)),
                       switch.spec = list(do.mix = FALSE))
  MSmodel_est<-FitML(spec = MSmodel, data = log_returns[1:window_size])
  
  for (i in 1:(size)){
    MSforecast = predict(MSmodel_est, nahead = n_ahead, newdata = log_returns[(1+window_size+i*n_ahead):(n_ahead+window_size+i*n_ahead)])
    setTxtProgressBar(pb, i)
    ms_h_step_forecast = c(ms_h_step_forecast, MSforecast$vol)
  }
  return(ms_h_step_forecast)
}