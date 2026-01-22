# install.packages("rstan") 
library(rstan)

setwd("C:\\Users\\saraa\\OneDrive\\Documents\\Research\\Sample_size_prediction\\Independent binomial-n reconstruction for a single observed row")

# Speed + caching
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


stan_data <- list(
  y = 71, # y_obs
  a = 1, b = 0.1,
  alpha = 2, beta = 2,
  alphastar = 1, betastar = 1,
  R_max = 80,
  N_max = 500
)

fit <- stan(
  file = "model.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  seed = 123
)

print(fit, pars = c("p","pstar","lambda","r_draw","n_draw","missing_cell"))
