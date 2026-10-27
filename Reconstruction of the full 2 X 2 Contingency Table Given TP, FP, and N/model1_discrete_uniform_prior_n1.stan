functions {
  // Beta-Binomial log PMF:
  // y ~ Binomial(n, p) and p ~ Beta(a, b)  =>  y ~ BetaBinomial(n, a, b)
  real beta_binom_lpmf_int(int y, int n, real a, real b) {
    return lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b);
  }
}
data {
  int<lower=1> N;//known total N
  int<lower=0> TP; //observed TP
  int<lower=0> TN;  //observed TN

  real<lower=0> a1;
  real<lower=0> b1;
  real<lower=0> a2;
  real<lower=0> b2;
}

transformed data {
  int L = (TP > 1) ? TP : 1; //L = max(1, TP)
  int U1 = N - 1;
  int U2 = N - TN;
  int U = (U1 < U2) ? U1 : U2; //U = min(N-1, N-TN)

  if (L > U)
    reject("No feasible n1: need TP <= n1 <= N-TN and 1<=n1<=N-1, but L>U.");

  real log_uniform = -log(N - 1.0);// log(1/(N-1))
}

parameters {
  real<lower=0, upper=1> p1;
  real<lower=0, upper=1> p2;
}

model {
  //priors 
  p1 ~ beta(a1, b1);
  p2 ~ beta(a2, b2);

  //enumerate n1 over feasible support only: n1 = L..U
  {
    vector[U - L + 1] log_w;

    for (n1 in L:U) {
      int n2 = N - n1;
      log_w[n1 - L + 1] =
        log_uniform
        + binomial_lpmf(TP | n1, p1)
        + binomial_lpmf(TN | n2, p2);
    }

    target += log_sum_exp(log_w);
  }
}

generated quantities {
  int n1_draw;
  int n2_draw;
  int FP_draw;
  int FN_draw;

  //draw n1 from p(n1 | p1, p2, data) each iteration
  {
    vector[U - L + 1] log_w;

    for (n1 in L:U) {
      int n2 = N - n1;
      log_w[n1 - L + 1] =
        log_uniform
        + binomial_lpmf(TP | n1, p1)
        + binomial_lpmf(TN | n2, p2);
    }

    n1_draw = (categorical_logit_rng(log_w) - 1) + L;  // map back to L..U
  }

  n2_draw = N - n1_draw;

  //full 2x2 reconstruction pieces
  FP_draw = n1_draw - TP;
  FN_draw = n2_draw - TN;
}
