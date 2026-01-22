library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_data <- list(
  N = 182,
  TP = 71,
  TN = 80,
  a1 = 1, b1 = 1,
  a2 = 1, b2 = 1
)

fit <- stan(
  file = "model1_discrete_uniform_prior_n1.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  seed = 123
)

print(fit, pars = c("p1","p2","n1_draw","n2_draw"))
