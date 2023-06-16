# Simulating log-return sequences using MS-GARCH package

library(fGarch)
library(dplyr)
library(tseries)
library(rugarch)
library(MSGARCH)

#### Simulating time series #### 

data_len = 71543

spec_arch = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,0)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                        fixed.pars=list(omega=0.0001, alpha1=0.99,
                                        shape=3))

spec_garch = ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                        fixed.pars=list(omega=0.0001, alpha1=0.05, beta1=0.90,
                                        shape=3))

spec_e_garch = ugarchspec(variance.model=list(model="eGARCH", garchOrder=c(1,1)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                        fixed.pars=list(omega=0.0001, alpha1=0.05, beta1=0.3,
                                        gamma1=0.05, shape=10))

spec_gjr_garch = ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)), 
                        mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="std", 
                        fixed.pars=list(omega=0.0001, alpha1=0.047, beta1=0.78,
                                        gamma1=0.33, shape=3))

# simulate the path
path_arch = ugarchpath(spec_arch, n.sim=data_len, n.start=1, m.sim=1)
path_garch = ugarchpath(spec_garch, n.sim=data_len, n.start=1, m.sim=1)
path_e_garch = ugarchpath(spec_e_garch, n.sim=data_len, n.start=1, m.sim=1)
path_gjr_garch = ugarchpath(spec_gjr_garch, n.sim=data_len, n.start=1, m.sim=1)

#Extract draws & condvols
sim_arch_draw = path_arch@path$seriesSim
sim_arch_condvol = path_arch@path$sigmaSim
sim_garch_draw = path_garch@path$seriesSim
sim_garch_condvol = path_garch@path$sigmaSim
sim_egarch_draw = path_e_garch@path$seriesSim
sim_egarch_condvol = path_e_garch@path$sigmaSim
sim_gjrgarch_draw = path_gjr_garch@path$seriesSim
sim_gjrgarch_condvol = path_gjr_garch@path$sigmaSim



# Set up MS-GARCH model with 2 regimes
model = "sGARCH"
distribution = "std"
regimes = 2

MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                     distribution.spec = list(distribution = c(distribution)), #'norm', 'std', 'ged'
                     switch.spec = list(do.mix = FALSE, K = regimes))

# Set fixed set of model parameters
#par = c( alpha0_1, alpha1_1, beta_1, nu_1, alpha0_2, alpha1_2, beta_2, nu_2, P_1_1)
par_ged <- c(0.0002, 0.08, 0.9, 1.4, 0.0003, 0.9, 0.01, 0.7, 0.85, 0.25) #GED
par_std <- c(0.0002, 0.08, 0.9, 2, 0.0003, 0.8, 0.01, 3, 0.25, 0.25) #STD
par_norm <- c(0.0002, 0.08, 0.9, 0.0003, 0.9, 0.01, 0.85, 0.25) #Normal

set.seed(1234)

# Set number of observations to simulate
n_sim <- 100

# Set initial state of Markov chain
init_state <- 1

# Simulate MS-GARCH sequence
#sim_norm <- simulate(object = MSGARCH, nsim = 1, nahead = 71444, nburn = 100, par = par_norm)
sim_std <- simulate(object = MSGARCH, nsim = 1, nahead = data_len, nburn = 100, par = par_std)
#sim_ged <- simulate(object = MSGARCH, nsim = 1, nahead = 71444, nburn = 100, par = par_ged)

#Create empty matrices
#cond_vol_norm <- matrix(nrow = (length(sim_norm$CondVol)/2), ncol = regimes+1)
cond_vol_std <- matrix(nrow = (length(sim_std$CondVol)/2), ncol = regimes+1)
#cond_vol_ged <- matrix(nrow = (length(sim_norm$CondVol)/2), ncol = regimes+1)

# STD
cond_vol_std[,1] = sim_std$CondVol[1:(length(sim_std$CondVol)/2)]
cond_vol_std[,2] = sim_std$CondVol[((length(sim_std$CondVol)/2)+1):length(sim_std$CondVol)]
cond_vol_std[,3] = sim_std$state

result_std = numeric(length(sim_std)/2)

for (i in 1:(length(sim_std$CondVol)/2)) {
  if (cond_vol_std[i, 3] == 1) {
    result_std[i] <- cond_vol_std[i, 1]
  } else if (cond_vol_std[i, 3] == 2) {
    result_std[i] <- cond_vol_std[i, 2]
  }
}

plot(sim_std$draw, type='l',col='blue')
lines(result_std, type = 'l')

# Export log_returns & known conditional volatility
#ARCH
write.table(sim_arch_draw, "./3_Simulated_garch/sim_arch_draws.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)
write.table(sim_arch_condvol, "./3_Simulated_garch/sim_arch_condvol.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)

#GARCH
write.table(sim_garch_draw, "./3_Simulated_garch/sim_garch_draws.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)
write.table(sim_garch_condvol, "./3_Simulated_garch/sim_garch_condvol.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)

#E-GARCH
write.table(sim_egarch_draw, "./3_Simulated_garch/sim_egarch_draws.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)
write.table(sim_egarch_condvol, "./3_Simulated_garch/sim_egarch_condvol.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)

#GJR-GARCH
write.table(sim_gjrgarch_draw, "./3_Simulated_garch/sim_gjrgarch_draws.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)
write.table(sim_gjrgarch_condvol, "./3_Simulated_garch/sim_gjrgarch_condvol.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)

#MS-GARCH
write.table(sim_std$draw, "./3_Simulated_garch/sim_msgarch_draws.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)
write.table(result_std, "./3_Simulated_garch/sim_msgarch_condvol.csv", sep = ",", row.names = FALSE, col.names = 'Col1', quote = TRUE)

plot(result_std)

'''

# Norm
cond_vol_norm[,1] = sim_norm$CondVol[1:(length(sim_norm$CondVol)/2)]
cond_vol_norm[,2] = sim_norm$CondVol[((length(sim_norm$CondVol)/2)+1):length(sim_norm$CondVol)]
cond_vol_norm[,3] = sim_norm$state

result_norm = numeric(length(sim_norm$CondVol)/2)

for (i in 1:(length(sim_norm$CondVol)/2)) {
  if (cond_vol_norm[i, 3] == 1) {
    result_norm[i] <- cond_vol_norm[i, 1]
  } else if (cond_vol_norm[i, 3] == 2) {
    result_norm[i] <- cond_vol_norm[i, 2]
  }
}

write.table(sim_norm$draw, "ms_norm_draws.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(result_norm, "ms_norm_CondVol.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


# GED
cond_vol_ged[,1] = sim_ged$CondVol[1:(length(sim_ged$CondVol)/2)]
cond_vol_ged[,2] = sim_ged$CondVol[((length(sim_ged$CondVol)/2)+1):length(sim_ged$CondVol)]
cond_vol_ged[,3] = sim_ged$state

result_ged = numeric(length(sim_ged$CondVol)/2)

for (i in 1:(length(sim_ged$CondVol)/2)) {
  if (cond_vol_ged[i, 3] == 1) {
    result_ged[i] <- cond_vol_ged[i, 1]
  } else if (cond_vol_ged[i, 3] == 2) {
    result_ged[i] <- cond_vol_ged[i, 2]
  }
}

write.table(sim_ged$draw, "ms_ged_draws.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(result_ged, "ms_ged_CondVol.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)

'''