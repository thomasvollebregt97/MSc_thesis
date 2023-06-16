filtered_ms = function(distr, log_returns_train){
  
  model = "sGARCH"
  distribution = distr
  regimes = 2
  
  MSGARCH = CreateSpec(variance.spec = list(model = c(model)),
                       distribution.spec = list(distribution = c(distribution)), #'norm', 'std', 'ged'
                       switch.spec = list(do.mix = FALSE, K = regimes))
  
  optimizers <- c("solnp", "nlminb", "Nelder-Mead")
  
  ms_fit = FitML(MSGARCH, data = log_returns_train)
  
  filt_var = Volatility(ms_fit)

  return(filt_var)
}