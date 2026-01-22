functions {
  // log p(FP | r, pstar, p) with n2 truncated to FP <= n2 <= N (known)
  // n2 ~ NegBin(r, pstar) truncated, and FP | n2 ~ Bin(n2, p)
  real log_pFP_given_r_knownN(int FP, int N, real p, real pstar, int r) {
    // truncation constant: P(FP <= n2 <= N | r, pstar)
    // = CDF(N) - CDF(FP-1)
    real log_trunc = log_diff_exp(
      neg_binomial_lcdf(N | r, pstar),
      neg_binomial_lcdf(FP - 1 | r, pstar)
    );
    
    vector[N - FP + 1] lterms;
    for (n2 in FP:N) {
      lterms[n2 - FP + 1] =
        neg_binomial_lpmf(n2 | r, pstar)
      + binomial_lpmf(FP | n2, p);
    }
    return log_sum_exp(lterms) - log_trunc;
  }
}

data {
  int<lower=0> FP;// observed false positives
  int<lower=1> N;// known total N (benign stratum cap)
  
  // hyperparameters
  real<lower=0> a;  // gamma shape for lambda
  real<lower=0> b;  // gamma rate for lambda
  real<lower=0> alpha;  // beta prior for p
  real<lower=0> beta;
  real<lower=0> alphastar; // beta prior for p*
    real<lower=0> betastar;
  
  int<lower=1> R_max;  // max r to sum over (use r>=1)
}

transformed data {
  if (FP > N)
    reject("Need FP <= N for truncation FP <= n2 <= N.");
}

parameters {
  real<lower=0, upper=1> p;// false positive rate
  real<lower=0, upper=1> pstar;   // negbin probability parameter
  real<lower=0> lambda; // rate for r ~ Poisson(lambda)
}

model {
  // Priors
  p      ~ beta(alpha, beta);
  pstar  ~ beta(alphastar, betastar);
  lambda ~ gamma(a, b); // Stan gamma(shape, rate)
  
  // Marginalize r and n2:
    // r ~ Poisson(lambda), summed over r=1..R_max (avoid degenerate r=0 NB)
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] =
        poisson_lpmf(r | lambda)
      + log_pFP_given_r_knownN(FP, N, p, pstar, r);
    }
    target += log_sum_exp(lr);
    }
}

generated quantities {
  int r_draw;
  int n2_draw;
  int missing_cell; //missing_cell = n2 - FP
  
  // draw r from its posterior discrete distribution
  {
    vector[R_max] lr;
    for (r in 1:R_max) {
      lr[r] =
        poisson_lpmf(r | lambda)
      + log_pFP_given_r_knownN(FP, N, p, pstar, r);
    }
    r_draw = categorical_logit_rng(lr); // 1..R_max
  }
  
  // draw n2 from p(n2 | r_draw, pstar, p, FP, N), truncated FP<=n2<=N
  {
    // log truncation constant
    real log_trunc = log_diff_exp(
      neg_binomial_lcdf(N | r_draw, pstar),
      neg_binomial_lcdf(FP - 1 | r_draw, pstar)
    );
    
    vector[N - FP + 1] ln;
    for (n2 in FP:N) {
      ln[n2 - FP + 1] =
        neg_binomial_lpmf(n2 | r_draw, pstar)
      + binomial_lpmf(FP | n2, p)
      - log_trunc;
    }
    n2_draw = (categorical_logit_rng(ln) - 1) + FP;
  }
  
  missing_cell = n2_draw - FP;
}
