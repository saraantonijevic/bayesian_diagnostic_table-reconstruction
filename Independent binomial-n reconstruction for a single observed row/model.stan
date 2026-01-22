functions {
  //log p(y | r, pstar, p, with n truncated to n>=y and summed to N_max)
  real log_py_given_r(real p, real pstar, int r, int y, int N_max) {
    // truncation constant P(n >= y | r, pstar) = P(n > y-1 | r, pstar)
    real log_trunc = neg_binomial_lccdf(y - 1 | r, pstar);

    vector[N_max - y + 1] lterms;
    for (n in y:N_max) {
      lterms[n - y + 1] =
        neg_binomial_lpmf(n | r, pstar) +
        binomial_lpmf(y | n, p);
    }
    return log_sum_exp(lterms) - log_trunc;
  }
}

data {
  int<lower=0> y;  //observed count (y_obs)

  // hyperparameters
  real<lower=0> a;   //gamma shape for lambda
  real<lower=0> b;  //gamma rate  for lambda
  real<lower=0> alpha; // beta prior for p
  real<lower=0> beta;
  real<lower=0> alphastar;// beta prior for pstar
  real<lower=0> betastar;

  // truncation controls for marginalization
  int<lower=1> R_max; // max r to sum over (r is forced >= 1)
  int<lower=y> N_max;  // max n to sum over (must be >= y)
}

parameters {
  real<lower=0, upper=1> p; // binomial success probability
  real<lower=0, upper=1> pstar;  // negbin success probability (Stan parameterization)
  real<lower=0> lambda; // Poisson rate for r
}

model {
  //priors
  p     ~ beta(alpha, beta);
  pstar ~ beta(alphastar, betastar);
  lambda ~ gamma(a, b);

  // Marginal likelihood:
  // r ~ Poisson(lambda) truncated to r=1..R_max
  // n ~ NegBin(r, pstar) truncated to n>=y, then y|n ~ Bin(n,p)
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] = poisson_lpmf(r | lambda) + log_py_given_r(p, pstar, r, y, N_max);
    }
    target += log_sum_exp(lr);
  }
}

generated quantities {
  int r_draw;
  int n_draw;
  int missing_cell;

  //draw r from its posterior discrete distribution given parameters (r = 1..R_max)
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] = poisson_lpmf(r | lambda) + log_py_given_r(p, pstar, r, y, N_max);
    }
    r_draw = categorical_logit_rng(lr); // returns 1..R_max
  }

  //draw n from posterior given r_draw and parameters (with truncation n>=y)
  {
    vector[N_max - y + 1] ln;
    real log_trunc = neg_binomial_lccdf(y - 1 | r_draw, pstar);

    for (n in y:N_max) {
      ln[n - y + 1] =
        neg_binomial_lpmf(n | r_draw, pstar) +
        binomial_lpmf(y | n, p) -
        log_trunc;
    }
    n_draw = (categorical_logit_rng(ln) - 1) + y;
  }

  missing_cell = n_draw - y;
}
