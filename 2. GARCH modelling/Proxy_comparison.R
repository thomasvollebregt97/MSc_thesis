# Analyse proxies: Realized variance vs Realized kernel
library(realized)
library("Metrics")

# Load simulations
sim_arch_draw = path_arch@path$seriesSim
sim_arch_condvol = path_arch@path$sigmaSim
sim_garch_draw = path_garch@path$seriesSim
sim_garch_condvol = path_garch@path$sigmaSim
sim_egarch_draw = path_e_garch@path$seriesSim
sim_egarch_condvol = path_e_garch@path$sigmaSim
sim_gjrgarch_draw = path_gjr_garch@path$seriesSim
sim_gjrgarch_condvol = path_gjr_garch@path$sigmaSim
sim_msgarch_draw = sim_std$draw
sim_msgarch_condvol = result_std

has_nan <- any(is.na(sim_arch_draw))

#### Estimate variance: realized variance ####

# Setting up vectors
h = 49
rv_arch = rep(0, length(sim_arch_draw)-h+1)
rv_garch = rep(0, length(sim_arch_draw)-h+1)
rv_egarch = rep(0, length(sim_arch_draw)-h+1)
rv_gjrgarch = rep(0, length(sim_arch_draw)-h+1)
rv_msgarch = rep(0, length(sim_arch_draw)-h+1)

# Calculate realized volatility for each sequence
for (i in 1:(length(sim_arch_draw)-h+1)){
  rv_arch[i] = sqrt(var(sim_arch_draw[i:(h+i)]))
}

for (i in 1:(length(sim_garch_draw)-h+1)){
  rv_garch[i] = sqrt(var(sim_garch_draw[i:(h+i)]))
}

for (i in 1:(length(sim_egarch_draw)-h+1)){
  rv_egarch[i] = sqrt(var(sim_egarch_draw[i:(h+i)]))
}

for (i in 1:(length(sim_gjrgarch_draw)-h+1)){
  rv_gjrgarch[i] = sqrt(var(sim_gjrgarch_draw[i:(h+i)]))
}

for (i in 1:(length(sim_msgarch_draw)-h+1)){
  rv_msgarch[i] = sqrt(var(sim_msgarch_draw[i:(h+i)]))
}


#### Calculate realized kernel for each sequence ####

# Setup vectors
rk_arch = rep(0, length(sim_arch_condvol))
rk_garch = rep(0, length(sim_arch_condvol))
rk_egarch = rep(0, length(sim_arch_condvol))
rk_gjrgarch = rep(0, length(sim_arch_condvol))
rk_msgarch = rep(0, length(sim_arch_condvol))

#Calculate realized kernel volatilities
for (i in 1:length(sim_arch_condvol)){
  rk_arch[i] = rv.kernel(x = sim_arch_draw[i:(h+i)])
}

for (i in 1:length(sim_arch_condvol)){
  rk_garch[i] = rv.kernel(x = sim_garch_draw[i:(h+i)])
}

for (i in 1:length(sim_arch_condvol)){
  rk_egarch[i] = rv.kernel(x = sim_egarch_draw[i:(h+i)])
}

for (i in 1:length(sim_arch_condvol)){
  rk_gjrgarch[i] = rv.kernel(x = sim_gjrgarch_draw[i:(h+i)])
}

for (i in 1:length(sim_arch_condvol)){
  rk_msgarch[i] = rv.kernel(x = sim_msgarch_draw[i:(h+i)])
}

plot(rk_arch, type='l',col='red')
lines(sim_arch_condvol[h+1:length(sim_arch_condvol)], type='l',col='blue')

lines(rk_arch, type='l',col='red')
plot(sim_arch_condvol[h+1:length(sim_arch_condvol)], type='l',col='blue')

plot(rk_garch^2, type='l',col='red')
lines(sim_garch_condvol[h+1:length(sim_arch_condvol)], type='l',col='blue')

plot(rk_gjrgarch, type='l',col='red')
lines(sim_gjrgarch_condvol[h+1:length(sim_arch_condvol)], type='l',col='blue')

# Make rk estimate and cond vol same size
sim_arch_condvol = sim_garch_condvol[-(1:h)]
sim_garch_condvol = sim_garch_condvol[-(1:h)]
sim_egarch_condvol = sim_garch_condvol[-(1:h)]
sim_gjrgarch_condvol = sim_garch_condvol[-(1:h)]
sim_msgarch_condvol = sim_garch_condvol[-(1:h)]

# Remove NaN indices from rk & cond_vol
nan_indices_rk_arch <- which(is.nan(rk_arch))
nan_indices_rk_garch <- which(is.nan(rk_garch))
nan_indices_rk_egarch <- which(is.nan(rk_egarch))
nan_indices_rk_gjrgarch <- which(is.nan(rk_gjrgarch))
nan_indices_rk_msgarch <- which(is.nan(rk_msgarch))

# Update both sequences
rk_arch <- rk_arch[-nan_indices_rk_arch]
sim_arch_condvol_rk <- sim_arch_condvol[-nan_indices_rk_arch]
rk_garch <- rk_garch[-nan_indices_rk_garch]
sim_garch_condvol_rk <- sim_garch_condvol[-nan_indices_rk_garch]
rk_egarch <- rk_egarch[-nan_indices_rk_egarch]
sim_egarch_condvol_rk <- sim_egarch_condvol[-nan_indices_rk_egarch]
rk_gjrgarch <- rk_gjrgarch[-nan_indices_rk_gjrgarch]
sim_gjrgarch_condvol_rk <- sim_gjrgarch_condvol[-nan_indices_rk_gjrgarch]
rk_msgarch <- rk_msgarch[-nan_indices_rk_msgarch]
sim_msgarch_condvol_rk <- sim_msgarch_condvol[-nan_indices_rk_msgarch]


# Calculate error differences of rv and rk wrt conditional volatility

# ARCH
mse_arch_rv = mse(rv_arch[-length(rv_arch)], sim_arch_condvol)
mse_arch_rk = mse(rk_arch, sim_arch_condvol_rk)
# GARCH
mse_garch_rv = mse(rv_garch[-length(rv_garch)], sim_garch_condvol)
mse_garch_rk = mse(rk_garch, sim_garch_condvol_rk)
#E-GARCH
mse_egarch_rv = mse(rv_egarch[-length(rv_egarch)], sim_egarch_condvol)
mse_egarch_rk = mse(rk_egarch[1:10], sim_egarch_condvol_rk[1:10])
#GJR-GARCH  
mse_gjrgarch_rv = mse(rv_gjrgarch[-length(rv_gjrgarch)], sim_gjrgarch_condvol)
mse_gjrgarch_rk = mse(rk_gjrgarch, sim_gjrgarch_condvol_rk)
#MS-GARCH  
mse_msgarch_rv = mse(rv_msgarch[-length(rv_msgarch)], sim_msgarch_condvol)
mse_msgarch_rk = mse(rk_msgarch, sim_msgarch_condvol_rk)


