#install.packages("rstan")
library(rstan)
setwd("C:\\Users\\saraa\\OneDrive\\Documents\\Research\\Sample_size_prediction\\Application\\Unknown N models")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_data <- list(
  TP = 71,
  a = 1, b = 0.1,
  alpha = 2, beta = 1,
  alphastar = 1, betastar = 1,
  R_max = 60,
  N_max = 250
)

fit<- stan(
  file = "model1_analysis_of_diseased_stratum.stan",
  data = stan_data,
  chains = 2,
  iter = 1000,
  seed = 123
)



print(fit, pars = c("p","pstar","lambda","r_draw","n1_draw","missing_cell"))

# 
# #some checks, bump N_max and see if posterior changes materially
# stan_data2 <- within(stan_data, { N_max <- 400 })
# fit2 <- stan(file="model1_analysis_of_diseased_stratum.stan",
#                   data=stan_data2, chains=2, iter=800, seed=123)
# 
# print(fit,  pars=c("p","pstar","lambda","n1_draw"))
# print(fit, pars=c("p","pstar","lambda","n1_draw"))
