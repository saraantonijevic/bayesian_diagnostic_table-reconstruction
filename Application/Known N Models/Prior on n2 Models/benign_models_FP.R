library(rstan)
setwd("C:\\Users\\saraa\\OneDrive\\Documents\\Research\\Sample_size_prediction\\Application\\Known N Models\\Prior on n2 Models")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_data <- list(
  FP = 28,
  N = 182, 
  a = 2, b = 1,
  alpha = 2, beta = 5,
  alphastar = 1, betastar = 50,
  R_max = 200
)

fit <- stan(
  file = "benign_knownN_FP_model.stan",
  data = stan_data,
  chains = 2,
  iter = 1000,
  seed = 123
)

print(fit, pars = c("p","pstar","lambda","r_draw","n2_draw","missing_cell"))
