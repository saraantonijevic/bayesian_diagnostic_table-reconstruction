data {
  int<lower=1> N;// known total N
  int<lower=0> TP;// observed TP
  int<lower=0> TN;// observed TN

  real<lower=0> a1;
  real<lower=0> b1;
  real<lower=0> a2;
  real<lower=0> b2;

  //gamma prior hyperparameters for lambda 
  real<lower=0> a_lambda;
  real<lower=0> b_lambda;
}

transformed data {
  //feasible n1 must satisfy: 1 <= n1 <= N-1, TP <= n1, and TN <= N-n1
  int L = (TP > 1) ? TP : 1;// max(1, TP)
  int U1 = N - 1;
  int U2 = N - TN;
  int U = (U1 < U2) ? U1 : U2;// min(N-1, N-TN)

  if (L > U)
    reject("No feasible n1 given TP,TN,N (L>U).");
}

parameters {
  real<lower=0, upper=1> p1;
  real<lower=0, upper=1> p2;
  real<lower=0> lambda;
}

model {
  //priors 
  p1 ~ beta(a1, b1);
  p2 ~ beta(a2, b2);
  lambda ~ gamma(a_lambda, b_lambda); //stan gamma(shape, rate)

  //n1 ~ Poisson(lambda) truncated to {1,...,N-1}
  //normalizing constant: Z_prior = P(1 <= n1 <= N-1 | lambda)
  //= CDF(N-1) - CDF(0)
  {
    real log_Zprior = log_diff_exp(poisson_lcdf(N - 1 | lambda),
                                   poisson_lcdf(0 | lambda));

    vector[U - L + 1] log_w;

    //sum only over feasible n1 values (same posterior; avoids impossible terms)
    for (n1 in L:U) {
      int n2 = N - n1;

      log_w[n1 - L + 1] =
        binomial_lpmf(TP | n1, p1)
        + binomial_lpmf(TN | n2, p2)
        + poisson_lpmf(n1 | lambda)
        - log_Zprior;
    }

    target += log_sum_exp(log_w);
  }
}

generated quantities {
  int n1_draw;
  int n2_draw;
  int FP_draw;
  int FN_draw;

  {
    real log_Zprior = log_diff_exp(poisson_lcdf(N - 1 | lambda),
                                   poisson_lcdf(0 | lambda));

    vector[U - L + 1] log_w;

    for (n1 in L:U) {
      int n2 = N - n1;

      log_w[n1 - L + 1] =
        binomial_lpmf(TP | n1, p1)
        + binomial_lpmf(TN | n2, p2)
        + poisson_lpmf(n1 | lambda)
        - log_Zprior;
    }

    n1_draw = (categorical_logit_rng(log_w) - 1) + L;
  }

  n2_draw = N - n1_draw;
  FP_draw = n1_draw - TP;
  FN_draw = n2_draw - TN;
}
