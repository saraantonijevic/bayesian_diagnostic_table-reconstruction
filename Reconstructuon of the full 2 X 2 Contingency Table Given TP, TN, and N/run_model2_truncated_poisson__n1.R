library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_data <- list(
  N = 182,
  TP = 71,
  TN = 80,
  a1 = 1, b1 = 1,
  a2 = 1, b2 = 1,
  a_lambda = 0.1,
  b_lambda = 0.1
)

fit <- stan(
  file = "model2_truncated_poisson_prior_on_n1.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  seed = 123
)

print(fit, pars = c("p1","p2","lambda","n1_draw","n2_draw","FP_draw","FN_draw"))
