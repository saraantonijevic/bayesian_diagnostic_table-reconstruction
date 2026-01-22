functions {
  // log p(TP | r, pstar, p) where:
    // n1 ~ NegBin(r, pstar) truncated to n1 >= TP
  // TP | n1 ~ Bin(n1, p)
  // We sum n1 from TP..N_max.
  real log_pTP_given_r(int TP, real p, real pstar, int r, int N_max) {
    // truncation constant: P(n1 >= TP | r, pstar) = P(n1 > TP-1)
    real log_trunc = neg_binomial_lccdf(TP - 1 | r, pstar);
    
    vector[N_max - TP + 1] lterms;
    for (n1 in TP:N_max) {
      lterms[n1 - TP + 1] =
        neg_binomial_lpmf(n1 | r, pstar)
      + binomial_lpmf(TP | n1, p);
    }
    return log_sum_exp(lterms) - log_trunc;
  }
}

data {
  int<lower=0> TP; // observed TP
  
  // hyperparameters
  real<lower=0> a; // gamma shape for lambda
  real<lower=0> b;  // gamma rate for lambda
  real<lower=0> alpha; // beta prior for p
  real<lower=0> beta;
  real<lower=0> alphastar; // beta prior for p*
    real<lower=0> betastar;
  
  // truncation cutoffs for marginalization
  int<lower=1> R_max;  // max r to sum over (use r>=1)
  int<lower=TP> N_max;   // max n1 to sum over (>= TP)
}

parameters {
  real<lower=0, upper=1> p; // sensitivity
  real<lower=0, upper=1> pstar; // negbin probability parameter
  real<lower=0> lambda; // rate for r ~ Poisson(lambda)
}

model {
  // Priors
  p      ~ beta(alpha, beta);
  pstar  ~ beta(alphastar, betastar);
  lambda ~ gamma(a, b); // Stan gamma(shape, rate)
  
  //Marginalize discrete r and n1:
    // r ~ Poisson(lambda) but we sum r=1..R_max (avoids degenerate NB at r=0)
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] = poisson_lpmf(r | lambda)
      + log_pTP_given_r(TP, p, pstar, r, N_max);
    }
    target += log_sum_exp(lr);
    }
}

generated quantities {
  int r_draw;
  int n1_draw;
  int missing_cell; //missing_cell = n1 - TP (FN in diseased stratum row)
  
  //Draw r from its posterior discrete distribution
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] = poisson_lpmf(r | lambda)
      + log_pTP_given_r(TP, p, pstar, r, N_max);
    }
    r_draw = categorical_logit_rng(lr); // returns 1..R_max
  }
  
  //Draw n1 from p(n1 | r_draw, pstar, p, TP) with truncation n1>=TP
  {
    vector[N_max - TP + 1] ln;
    real log_trunc = neg_binomial_lccdf(TP - 1 | r_draw, pstar);
    
    for (n1 in TP:N_max) {
      ln[n1 - TP + 1] =
        neg_binomial_lpmf(n1 | r_draw, pstar)
      + binomial_lpmf(TP | n1, p)
      - log_trunc;
    }
    n1_draw = (categorical_logit_rng(ln) - 1) + TP;
  }
  
  missing_cell = n1_draw - TP;
}
