### GARCH Model evaluations

library(fGarch)
library(dplyr)
library(tseries)
library(rugarch)
library(MSGARCH) 
source("forecast_eval.R")
source("forecast_eval_arch.R")
source("forecast_eval_ms.R")
source("forecast_eval_ms_plus.R") #For generating 1 extra filtered observation
source("rolling_forecast_ms.R")
source("rolling_forecast_ms_4f.R")
source("filtered_ms.R")
source("produce_forecast_and_filter_ms.R")
source("generate_matrix_results.R")


# Importing the data
log_returns = read.csv("log_returns_spain.csv", header = TRUE)
log_returns = log_returns$Price..EUR.MWhe.


#Train set
h = 20000 #(~30%) Forecast horizon, choose a multiple of the refit.every variable
ref = 10000
log_returns_train = log_returns[1:(length(log_returns)-h)]
log_returns_test = log_returns[(length(log_returns)-h+1):(length(log_returns))]


#### Create & evaluate forecasts (1-step ahead rolling forecasts) ####

#sARCH (5 lags chosen...)
error_a_norm = forecast_eval_arch(log_returns, "sGARCH", "norm", h = h, refit_every = ref)
error_a_std = forecast_eval_arch(log_returns, "sGARCH", "std", h = h, refit_every = ref)
error_a_ged = forecast_eval_arch(log_returns, "sGARCH", "ged", h = h, refit_every = ref) # Use: solver = "gosolnp"

#sGARCH
error_s_norm = forecast_eval(log_returns, "sGARCH", "norm", h = h, refit_every = ref)
error_s_std = forecast_eval(log_returns, "sGARCH", "std", h = h, refit_every = ref)
error_s_ged = forecast_eval(log_returns, "sGARCH", "ged", h = h, refit_every = ref)

#gjrGARCH
error_g_norm = forecast_eval(log_returns, "gjrGARCH", "norm", h = h, refit_every = ref)
error_g_std = forecast_eval(log_returns, "gjrGARCH", "std", h = h, refit_every = ref)
error_g_ged = forecast_eval(log_returns, "gjrGARCH", "ged", h = h, refit_every = ref)

#### eGARCH - Does not seem to work: explanation in report why? ####
error_e_norm = forecast_eval(log_returns, "eGARCH", "norm", h=h, refit_every = ref)
error_e_std = forecast_eval(log_returns, "eGARCH", "std", h=h,  refit_every = ref)
error_e_ged = forecast_eval(log_returns, "eGARCH", "ged", h=h, refit_every = ref)


### Training set for rolling forecast ms-garch
train_1 = log_returns_train
train_2 = c(log_returns_train[(ref+1):length(log_returns_train)],log_returns_test[1:ref])
train_2_std = c(log_returns_train[(ref+1):length(log_returns_train)],log_returns_test[1:3500],log_returns_test[3750:5000],log_returns_test[(ref/2):ref])
#train_3 = c(log_returns_train[(2*ref+1):length(log_returns_train)], log_returns_test[2:(2*ref)]) #Doesn't converge with 1:2*ref...
#train_4 = c(log_returns_train[(3*ref+1):length(log_returns_train)], log_returns_test[10:(3*ref)])

test_1 = log_returns_test[1:ref]
test_2 = log_returns_test[(ref+1):(2*ref)]
#test_3 = log_returns_test[(2*ref+1):(3*ref)]
#test_4 = log_returns_test[(3*ref+1):(4*ref)]

rv_test = error_a_norm[,2]

#MSGARCH (based on 20.000 test set and every 5000 a refit)
error_ms_norm = rolling_forecast_ms("norm", train_1, test_1, train_2, test_2, rv_test)
error_ms_std = rolling_forecast_ms("std", train_1, test_1, train_2_std, test_2, rv_test)
error_ms_ged = rolling_forecast_ms("ged", train_1, test_1, train_2, test_2, rv_test)


####  ####

# Export forecasts
s = 5 # Amount of models
l = 3 # Amount of distributions
matrix_predictions = matrix(nrow = h, ncol = s*l)

col_names_pred = c("ARCH-Norm","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
              "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
              "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
colnames(matrix_results) = col_names_pred

print(length(error_a_norm[,1]))

#ARCH
matrix_predictions[,1] = error_a_norm[,1]
matrix_predictions[,2] = error_a_std[,1]
matrix_predictions[,3] = error_a_ged[,1]
#GARCH
matrix_predictions[,4] = error_s_norm[,1]
matrix_predictions[,5] = error_s_std[,1]
matrix_predictions[,6] = error_s_ged[,1]
#E-GARCH
matrix_predictions[,7] = error_e_norm[,1]
matrix_predictions[,8] = error_e_std[,1]
matrix_predictions[,9] = error_e_ged[,1]
#GJR-GARCH
matrix_predictions[,10] = error_g_norm[,1]
matrix_predictions[,11] = error_g_std[,1]
matrix_predictions[,12] = error_g_ged[,1]
#MS-GARCH
matrix_predictions[,13] = error_ms_norm[,1]
matrix_predictions[,14] = error_ms_std[,1]
matrix_predictions[,15] = error_ms_ged[,1]

write.table(matrix_predictions, "./1_Predicted_garch/matrix_predictions.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#### Export filtered sequence ####

s = 2 # Amount of models
l = 2 # Amount of distributions
matrix_filtered = matrix(nrow = length(log_returns), ncol = s*l)

col_names_filt = c("E-GARCH-Std","E-GARCH-GED","MS-GARCH-Std","MS-GARCH-GED")
colnames(matrix_filtered) = col_names_filt

# MS-GARCH std/ged
filt_ms_std = produce_forecast_and_filter_ms("std", log_returns_train, log_returns_test)
filt_ms_ged = produce_forecast_and_filter_ms("ged", log_returns_train, log_returns_test)


#E-GARCH: STD
spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = "std")
fit <- ugarchfit(spec, data = log_returns_train)
volatility <- as.numeric(sigma(fit))
export_gjr_std = c(volatility, error_e_std[,1])

#E-GARCH: GED
spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = "ged")
fit <- ugarchfit(spec, data = log_returns_train)
volatility <- as.numeric(sigma(fit))
export_gjr_ged = c(volatility, error_e_ged[,1])

matrix_filtered[,1] = filt_ms_std
matrix_filtered[,2] = filt_ms_ged
matrix_filtered[,3] = export_gjr_std
matrix_filtered[,4] = export_gjr_ged

# Export matrix
write.table(matrix_predictions, "./2_Filtered_garch/matrix_predictions.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)



#### Simulating time series #### 

spec_garch = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                        fixed.pars=list(mu=0.001,omega=0.00001, alpha1=0.05, beta1=0.90,
                                        shape=4,skew=2))

spec_garch = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)), 
                mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                fixed.pars=list(mu=0.001,omega=0.00001, alpha1=0.05, beta1=0.90,
                                shape=4,skew=2))
# simulate the path
path.sgarch = ugarchpath(spec, n.sim=3000, n.start=1, m.sim=1)



# Coeff: sGARCH
model = "sGARCH"
distr = 'std'
spec <- ugarchspec(variance.model = list(model = model, garchOrder = c(1, 0)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = distr)

fit <- ugarchfit(spec, data = log_returns_train)
print(fit@fit$coef)

# Coeff: MS-GARCH
model = "sGARCH"
distr = 'std'
regimes = 2

MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                     distribution.spec = list(distribution = c(distr)), #'norm', 'std', 'ged'
                     switch.spec = list(do.mix = FALSE, K = regimes))

ms_fit = FitML(MSGARCH, data = log_returns_train)
print(ms_fit$par)


#Comparing forecasts using plots
y1 = error_a_ged[,1]
y2 = error_e_ged[,2]
y3 = error_s_norm[,1]

plot(y1, type="l", col="blue", lwd=1, xlab="x", ylab="y")
lines(y2, type="l", col="red", lwd=1)
lines(y3, type="l", col="green", lwd=1)
legend("topright", legend=c("y1", "y2", "y3"), col=c("blue", "red","green"), lwd=2)



##### MS-GARCH filtered estimates
model = "sGARCH"
distribution = "ged"
#data = c(log_returns_train[1000:length(log_returns_train)],log_returns_test[1:1000])
data = log_returns_train
regimes = 2

MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                     distribution.spec = list(distribution = c(distribution)), #'norm', 'std', 'ged'
                     switch.spec = list(do.mix = FALSE, K = regimes))
ms_fit = FitML(MSGARCH, data = data)

filt_var = Volatility(ms_fit, newdata = log_returns_test)
filt_var = filt_var[(length(log_returns_train)):length(log_returns)]
  
plot(log_returns, type="l", col="blue", lwd=1, xlab="x", ylab="y")
lines(filt_var, type="l", col="red", lwd=1)

print(sum(is.nan(filt_var)))



#### MS garch evaluation ####
#The filtered volatility can be considered the 1 step ahead forecast

#Training set 1
train_1 = log_returns_train
train_2 = c(log_returns_train[(ref+1):length(log_returns_train)], log_returns_test[1:(1*ref)])
train_3 = c(log_returns_train[(2*ref+1):length(log_returns_train)], log_returns_test[2:(2*ref)]) #Doesn't converge with 1:2*ref...
train_4 = c(log_returns_train[(3*ref+1):length(log_returns_train)], log_returns_test[1:(3*ref)])

test_1 = log_returns_test[1:ref]
test_2 = log_returns_test[(ref+1):(2*ref)]
test_3 = log_returns_test[(2*ref+1):(3*ref)]
test_4 = log_returns_test[(3*ref+1):(4*ref)]

# Normal
forecast_1_norm = forecast_eval_ms_plus("norm", train_1, test_1) #for generating 1 extra filtered observation
forecast_2_norm = forecast_eval_ms("norm", train_2, test_2)
forecast_3_norm = forecast_eval_ms("norm", train_3, test_3)
forecast_4_norm = forecast_eval_ms("norm", train_4, test_4)

rolling_forecast_ms_norm = c(forecast_1_norm, forecast_2_norm, forecast_3_norm, forecast_4_norm) #Refitted every 5000 observations
rolling_forecast_ms_norm = head(rolling_forecast_ms_norm, -1) #Export this rolling forecast
plot(rolling_forecast_ms_norm, type = 'l')

# Student-t
forecast_1_std = forecast_eval_ms_plus("std", train_1, test_1) #for generating 1 extra filtered observation
forecast_2_std = forecast_eval_ms("std", train_2, test_2)
forecast_3_std = forecast_eval_ms("std", train_3, test_3)
forecast_4_std = forecast_eval_ms("std", train_4, test_4)

rolling_forecast_ms_std = c(forecast_1_std, forecast_2_std, forecast_3_std, forecast_4_std) #Refitted every 5000 observations
rolling_forecast_ms_std = head(rolling_forecast_ms_std, -1) #Export this rolling forecast
plot(rolling_forecast_ms_std, type='l')
sum(is.na(rolling_forecast_ms_std))

# GED
forecast_1_ged = forecast_eval_ms_plus("ged", train_1, test_1) #for generating 1 extra filtered observation
forecast_2_ged = forecast_eval_ms("ged", train_2, test_2)
forecast_3_ged = forecast_eval_ms("ged", train_3, test_3)
forecast_4_ged = forecast_eval_ms("ged", train_4, test_4)

rolling_forecast_ms_ged = c(forecast_1_ged, forecast_2_ged, forecast_3_ged, forecast_4_ged) #Refitted every 5000 observations
rolling_forecast_ms_ged = head(rolling_forecast_ms_ged, -1)

plot(rolling_forecast_ms_ged, type ='l')
lines(rolling_forecast_ms_std, , type="l", col="blue", lwd=1)
lines(rv[51445:71444], type="l", col="red", lwd=1)



#### Testing optimizers for train and test set 3: because it does not converge ####

#Training set 1
train_1 = log_returns_train
train_2 = c(log_returns_train[(ref+1):length(log_returns_train)], log_returns_test[1:(1*ref)])

test_1 = log_returns_test[1:ref]
test_2 = log_returns_test[(ref+1):(2*ref)]



model = "sGARCH"
distribution = "std"
regimes = 2

MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                     distribution.spec = list(distribution = c(distribution)), #'norm', 'std', 'ged'
                     switch.spec = list(do.mix = FALSE, K = regimes))

optimizers <- c("solnp", "nlminb", "Nelder-Mead")

start <- list(
  "alpha[1,1]" = 0.05,
  "alpha[2,2]" = 0.1,
  "beta[1,1]" = 0.7,
  "beta[2,2]" = 0.5
)

ms_fit = FitML(MSGARCH, data = train_2)

filt_var = Volatility(ms_fit, newdata = test_2)

filt_var = filt_var[(length(train_1)+1):(length(train_1)+length(test_1))]


