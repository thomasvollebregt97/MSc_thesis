###### MS-GARCH tests ####

n_ahead = 12
size = 1000 #Maximum: round(h / n_ahead), i.e. round(20.000 / 12) 
buf = 48000
ms_12_step_forecast = numeric()
distr = "norm"

pb <- txtProgressBar(min = 0, max = size, style = 3)

l = length(log_returns[(buf+(i*n_ahead)):(i*n_ahead+(length(log_returns)-h))])

for (i in 1:size){
  MSmodel = CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH")),
                       distribution.spec = list(distribution = c(distr, distr)),
                       switch.spec = list(do.mix = FALSE))
  MSmodel_est<-FitML(spec = MSmodel, data = log_returns[(buf+(i*n_ahead)):(i*n_ahead+(length(log_returns)-h))])
  MSforecast <- predict(MSmodel_est, nahead = n_ahead)
  ms_12_step_forecast <- c(ms_12_step_forecast, MSforecast$vol)
  setTxtProgressBar(pb, i)
}

# Test with plots
plot(ms_12_step_forecast, type='l') #Access the rolled forecast
lines(rv[(length(rv)-h):length(rv)], type='l', col='blue')

plot(rv[(length(rv)-h):length(rv)], type='l', col='blue')
lines(ms_12_step_forecast, type='l')
