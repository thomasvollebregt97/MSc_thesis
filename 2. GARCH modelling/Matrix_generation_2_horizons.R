Matrix_generation_2_horizons = function(h1, h2, window_size, log_returns){
  ##### N-step ahead forecasts GARCH-type models: feature engineering ####
  
  ##### h = 12 ####
  
  # ARCH model
  vec_arch_norm_12 = produce_vector_forecasts_arch("sGARCH", "norm", window_size, log_returns, h1)
  vec_arch_std_12 = produce_vector_forecasts_arch("sGARCH", "std", window_size, log_returns, h1)
  vec_arch_ged_12 = produce_vector_forecasts_arch("sGARCH", "ged", window_size, log_returns, h1)
  
  # GARCH model
  vec_garch_norm_12 = produce_vector_forecasts("sGARCH", "norm", window_size, log_returns, h1)
  vec_garch_std_12 = produce_vector_forecasts("sGARCH", "std", window_size, log_returns, h1)
  vec_garch_ged_12 = produce_vector_forecasts("sGARCH", "ged", window_size, log_returns, h1)
  
  # E-GARCH model
  vec_egarch_norm_12 = produce_vector_forecasts("eGARCH", "norm", window_size, log_returns, h1)
  vec_egarch_std_12 = produce_vector_forecasts("eGARCH", "std", window_size, log_returns, h1)
  vec_egarch_ged_12 = produce_vector_forecasts("eGARCH", "ged", window_size, log_returns, h1)
  
  # GJR-GARCH model
  vec_gjrgarch_norm_12 = produce_vector_forecasts("gjrGARCH", "norm", window_size, log_returns, h1)
  vec_gjrgarch_std_12 = produce_vector_forecasts("gjrGARCH", "std", window_size, log_returns, h1)
  vec_gjrgarch_ged_12 = produce_vector_forecasts("gjrGARCH", "ged", window_size, log_returns, h1)
  
  # MS-GARCH model (t.b.c.)
  vec_msgarch_norm_12 = produce_vector_forecasts_ms("norm", window_size, log_returns, h1)
  vec_msgarch_norm_12 = produce_vector_forecasts("norm", window_size, log_returns, h1)
  vec_msgarch_std_12 = produce_vector_forecasts("std", window_size, log_returns, h1)
  
  #### h = 24 ####
  
  # ARCH model
  vec_arch_norm_24 = produce_vector_forecasts_arch("sGARCH", "norm", window_size, log_returns, h2)
  vec_arch_std_24 = produce_vector_forecasts_arch("sGARCH", "std", window_size, log_returns, h2)
  vec_arch_ged_24 = produce_vector_forecasts_arch("sGARCH", "ged", window_size, log_returns, h2)
  
  # GARCH model
  vec_garch_norm_24 = produce_vector_forecasts("sGARCH", "norm", window_size, log_returns, h2)
  vec_garch_std_24 = produce_vector_forecasts("sGARCH", "std", window_size, log_returns, h2)
  vec_garch_ged_24 = produce_vector_forecasts("sGARCH", "ged", window_size, log_returns, h2)
  
  # E-GARCH model
  vec_egarch_norm_24 = produce_vector_forecasts("eGARCH", "norm", window_size, log_returns, h2)
  vec_egarch_std_24 = produce_vector_forecasts("eGARCH", "std", window_size, log_returns, h2)
  vec_egarch_ged_24 = produce_vector_forecasts("eGARCH", "ged", window_size, log_returns, h2)
  
  # GJR-GARCH model
  vec_gjrgarch_norm_24 = produce_vector_forecasts("gjrGARCH", "norm", window_size, log_returns, h2)
  vec_gjrgarch_std_24 = produce_vector_forecasts("gjrGARCH", "std", window_size, log_returns, h2)
  vec_gjrgarch_ged_24 = produce_vector_forecasts("gjrGARCH", "ged", window_size, log_returns, h2)
  
  # MS-GARCH model
  vec_msgarch_norm_24 = produce_vector_forecasts("norm", window_size, log_returns, h2)
  vec_msgarch_norm_24 = produce_vector_forecasts("norm", window_size, log_returns, h2)
  vec_msgarch_std_24 = produce_vector_forecasts("std", window_size, log_returns, h2)
  
  #### Export all features: ####
  
  # h = 12
  s = 5 # Amount of models
  l = 3 # Amount of distributions
  EP_feature_matrix_12 = matrix(nrow = length(vec_arch_norm_12), ncol = s*l)
  
  col_names_pred = c("ARCH-Norm_12","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                     "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                     "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
  colnames(EP_pred_matrix_12) = col_names_pred
  
  #ARCH
  EP_feature_matrix_12[,1] = vec_arch_norm_12
  EP_feature_matrix_12[,2] = vec_arch_std_12
  EP_feature_matrix_12[,3] = vec_arch_ged_12
  #GARCH
  EP_feature_matrix_12[,4] = vec_garch_norm_12
  EP_feature_matrix_12[,5] = vec_garch_std_12
  EP_feature_matrix_12[,6] = vec_garch_ged_12
  #E-GARCH
  EP_feature_matrix_12[,7] = vec_egarch_norm_12
  EP_feature_matrix_12[,8] = vec_egarch_std_12
  EP_feature_matrix_12[,9] = vec_egarch_ged_12
  #GJR-GARCH
  EP_feature_matrix_12[,10] = vec_gjrgarch_norm_12
  EP_feature_matrix_12[,11] = vec_gjrgarch_std_12
  EP_feature_matrix_12[,12] = vec_gjrgarch_ged_12
  #MS-GARCH
  EP_feature_matrix_12[,13] = vec_gjrgarch_norm_12 #vec_msgarch_norm_12
  EP_feature_matrix_12[,14] = vec_gjrgarch_norm_12 #vec_msgarch_std_12
  EP_feature_matrix_12[,15] = vec_gjrgarch_norm_12 #vec_msgarch_ged_12
  
  # h = 24
  s = 5 # Amount of models
  l = 3 # Amount of distributions
  EP_pred_matrix_24 = matrix(nrow = length(vec_arch_norm_24), ncol = s*l)
  
  col_names_pred = c("ARCH-Norm_24","ARCH-Std","ARCH-GED","GARCH-Norm","GARCH-Std","GARCH-GED",
                     "E-GARCH-Norm","E-GARCH-Std","E-GARCH-GED","GJR-GARCH-Norm","GJR-GARCH-Std","GJR-GARCH-GED",
                     "MS-GARCH-Norm","MS-GARCH-Std","MS-GARCH-GED")
  colnames(EP_pred_matrix_24) = col_names_pred
  
  #ARCH
  EP_pred_matrix_24[,1] = vec_arch_norm_24
  EP_pred_matrix_24[,2] = vec_arch_std_24
  EP_pred_matrix_24[,3] = vec_arch_ged_24
  #GARCH
  EP_pred_matrix_24[,4] = vec_garch_norm_24
  EP_pred_matrix_24[,5] = vec_garch_std_24
  EP_pred_matrix_24[,6] = vec_garch_ged_24
  #E-GARCH
  EP_pred_matrix_24[,7] = vec_egarch_norm_24
  EP_pred_matrix_24[,8] = vec_egarch_std_24
  EP_pred_matrix_24[,9] = vec_egarch_ged_24
  #GJR-GARCH
  EP_pred_matrix_24[,10] = vec_gjrgarch_norm_24
  EP_pred_matrix_24[,11] = vec_gjrgarch_std_24
  EP_pred_matrix_24[,12] = vec_gjrgarch_ged_24
  #MS-GARCH
  EP_pred_matrix_24[,13] = vec_gjrgarch_norm_24 #vec_msgarch_norm_12
  EP_pred_matrix_24[,14] = vec_gjrgarch_norm_24 #vec_msgarch_std_12
  EP_pred_matrix_24[,15] = vec_gjrgarch_norm_24 #vec_msgarch_ged_12
  
  return()
}