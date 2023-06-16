# Generate filtered sequence from simulations
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
source('forecast_eval.R')
source('generate_filter_arch.R')
source('generate_filter.R')
source("produce_forecast_and_filter_ms.R") 


# Importing the simulated sequences
sim_arch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_arch_draws.csv", header = TRUE)$Col1
sim_garch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_garch_draws.csv", header = TRUE)$Col1
sim_egarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_egarch_draws.csv", header = TRUE)$Col1
sim_gjrgarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_gjrgarch_draws.csv", header = TRUE)$Col1
sim_msgarch_draw = read.csv("./2_Simulated_sequences/Simulated_sequences/sim_msgarch_draws.csv", header = TRUE)$Col1



# Filter volatilities from ARCH
h = 20000 #(~30%) Forecast horizon, choose a multiple of the refit.every variable
ref = 10000

#### ARCH: Std ####
sim_arch_train = sim_arch_draw[1:(length(sim_arch_draw)-h)]
sim_arch_test = sim_arch_draw[(length(sim_arch_draw)-h+1):(length(sim_arch_draw))]

# 1-step ahead features
filter_arch = generate_filter_arch(sim_arch_draw, sim_arch_train, h, ref) #Obs test
filter_garch = generate_filter(sim_arch_draw, sim_arch_train, h, ref, 'sGARCH')
filter_egarch = generate_filter(sim_arch_draw, sim_arch_train, h, ref, 'eGARCH')
filter_egarch[filter_egarch>3] = 3 #count = sum(filter_egarch>3)
filter_gjrgarch = generate_filter(sim_arch_draw, sim_arch_train, h, ref, 'gjrGARCH')
filter_msgarch = produce_forecast_and_filter_ms('std', sim_arch_train, sim_arch_test)

# Export features based on arch simulated sequence
write.table(filter_arch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch, "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)

# Export predictions based on arch simulated sequence
write.table(filter_arch[(length(filter_arch)-h):length(filter_arch)], "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_ARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch[(length(filter_arch)-h):length(filter_arch)], "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch[(length(filter_arch)-h):length(filter_arch)], "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_EGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch[(length(filter_arch)-h):length(filter_arch)], "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_GJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch[(length(filter_arch)-h):length(filter_arch)], "./2_Simulated_sequences/Features_simulations/ARCH sequence/ARCH_1_MSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)



#######################################################  GARCH: Std  ###################################################
sim_garch_train = sim_garch_draw[1:(length(sim_garch_draw)-h)]
sim_garch_test = sim_garch_draw[(length(sim_garch_draw)-h+1):(length(sim_garch_draw))]

filter_arch = generate_filter_arch(sim_garch_draw, sim_garch_train, h, ref) #Obs test
filter_garch = generate_filter(sim_garch_draw, sim_garch_train, h, ref, 'sGARCH')
filter_egarch = generate_filter(sim_garch_draw, sim_garch_train, h, ref, 'eGARCH')
filter_gjrgarch = generate_filter(sim_garch_draw, sim_garch_train, h, ref, 'gjrGARCH')
filter_msgarch = produce_forecast_and_filter_ms('std', sim_garch_train, sim_garch_test)

plot(filter_egarch, type='l')
lines(filter_gjrgarch, type='l', col='blue')
lines(filter_garch, type='l', col='red')
lines(filter_arch, type='l', col='green')
lines(filter_msgarch, type='l', col='grey')

# Export filters based on arch simulated sequence
write.table(filter_arch, "./4_Filtered_simulations/GARCH sequence/GARCH_filtARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch, "./4_Filtered_simulations/GARCH sequence/GARCH_filtGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch, "./4_Filtered_simulations/GARCH sequence/GARCH_filtEGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch, "./4_Filtered_simulations/GARCH sequence/GARCH_filtGJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch, "./4_Filtered_simulations/GARCH sequence/GARCH_filtMSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  E-GARCH: Std  ###################################################
sim_egarch_train = sim_egarch_draw[1:(length(sim_egarch_draw)-h)]
sim_egarch_test = sim_egarch_draw[(length(sim_egarch_draw)-h+1):(length(sim_egarch_draw))]

filter_arch = generate_filter_arch(sim_egarch_draw, sim_egarch_train, h, ref) #Obs test
filter_garch = generate_filter(sim_egarch_draw, sim_egarch_train, h, ref, 'sGARCH')
filter_egarch = generate_filter(sim_egarch_draw, sim_egarch_train, h, ref, 'eGARCH')
filter_gjrgarch = generate_filter(sim_egarch_draw, sim_egarch_train, h, ref, 'gjrGARCH')
filter_msgarch = produce_forecast_and_filter_ms('std', sim_egarch_train, sim_egarch_test)

plot(filter_egarch, type='l')
lines(filter_gjrgarch, type='l', col='blue')
lines(filter_garch, type='l', col='red')
lines(filter_arch, type='l', col='green')
lines(filter_msgarch, type='l', col='grey')

# Export filters based on arch simulated sequence
write.table(filter_arch, "./4_Filtered_simulations/E-GARCH sequence/E-GARCH_filtARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch, "./4_Filtered_simulations/E-GARCH sequence/E-GARCH_filtGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch, "./4_Filtered_simulations/E-GARCH sequence/E-GARCH_filtEGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch, "./4_Filtered_simulations/E-GARCH sequence/E-GARCH_filtGJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch, "./4_Filtered_simulations/E-GARCH sequence/E-GARCH_filtMSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  GJR-GARCH: Std  ###################################################
sim_gjrgarch_train = sim_gjrgarch_draw[1:(length(sim_gjrgarch_draw)-h)]
sim_gjrgarch_test = sim_gjrgarch_draw[(length(sim_gjrgarch_draw)-h+1):(length(sim_gjrgarch_draw))]

filter_arch = generate_filter_arch(sim_gjrgarch_draw, sim_gjrgarch_train, h, ref) #Obs test
filter_garch = generate_filter(sim_gjrgarch_draw, sim_gjrgarch_train, h, ref, 'sGARCH')
filter_egarch = generate_filter(sim_gjrgarch_draw, sim_gjrgarch_train, h, ref, 'eGARCH')
filter_gjrgarch = generate_filter(sim_gjrgarch_draw, sim_gjrgarch_train, h, ref, 'gjrGARCH')
filter_msgarch = produce_forecast_and_filter_ms('std', sim_gjrgarch_train, sim_gjrgarch_test)
filter_msgarch[filter_msgarch>3] = 3

plot(filter_gjrgarch, type='l')
lines(filter_egarch, type='l', col='blue')
lines(filter_garch, type='l', col='red')
lines(filter_arch, type='l', col='green')
plot(filter_msgarch, type='l', col='grey')

# Export filters based on arch simulated sequence
write.table(filter_arch, "./4_Filtered_simulations/GJR-GARCH sequence/GJR-GARCH_filtARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch, "./4_Filtered_simulations/GJR-GARCH sequence/GJR-GARCH_filtGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch, "./4_Filtered_simulations/GJR-GARCH sequence/GJR-GARCH_filtEGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch, "./4_Filtered_simulations/GJR-GARCH sequence/GJR-GARCH_filtGJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch, "./4_Filtered_simulations/GJR-GARCH sequence/GJR-GARCH_filtMSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


#######################################################  MS-GARCH: Std  ###################################################
sim_msgarch_train = sim_msgarch_draw[1:(length(sim_msgarch_draw)-h)]
sim_msgarch_test = sim_msgarch_draw[(length(sim_msgarch_draw)-h+1):(length(sim_msgarch_draw))]

filter_arch_ms = generate_filter_arch(sim_msgarch_draw, sim_msgarch_train, h, ref) #Obs test
filter_garch_ms = generate_filter(sim_msgarch_draw, sim_msgarch_train, h, ref, 'sGARCH')
filter_egarch_ms = generate_filter(sim_msgarch_draw, sim_msgarch_train, h, ref, 'eGARCH')
filter_egarch_ms[filter_egarch_ms > 5] = 5
filter_gjrgarch_ms = generate_filter(sim_msgarch_draw, sim_msgarch_train, h, ref, 'gjrGARCH')
filter_msgarch_ms = produce_forecast_and_filter_ms('std', sim_msgarch_train, sim_msgarch_test)

plot(sim_msgarch_train, type='l')
lines(filter_gjrgarch_ms, type='l', col='blue')

plot(filter_garch_ms, type='l')
lines(filter_egarch_ms, type='l', col='red')
lines(filter_arch_ms, type='l', col='green')
lines(filter_msgarch_ms, type='l', col='grey')

# Export filters based on arch simulated sequence
write.table(filter_arch_ms, "./4_Filtered_simulations/MS-GARCH sequence/MS-GARCH_filtARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_garch_ms, "./4_Filtered_simulations/MS-GARCH sequence/MS-GARCH_filtGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_egarch_ms, "./4_Filtered_simulations/MS-GARCH sequence/MS-GARCH_filtEGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_gjrgarch_ms, "./4_Filtered_simulations/MS-GARCH sequence/MS-GARCH_filtGJRGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)
write.table(filter_msgarch_ms, "./4_Filtered_simulations/MS-GARCH sequence/MS-GARCH_filtMSGARCH.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = TRUE)


