data {
  int<lower=1> N;// known total N
  int<lower=0> TP;// observed TP
  int<lower=0> TN; // observed TN
  
  real<lower=0> a1;
  real<lower=0> b1;
  real<lower=0> a2;
  real<lower=0> b2;
  
  real<lower=0> a3;
  real<lower=0> b3;
  
  //gamma prior hyperparameters for r (shape-rate), matching WinBUGS dgamma(ar, br)
  real<lower=0> a_r;
  real<lower=0> b_r;
}

transformed data {
  //feasible n1 must satisfy: 1 <= n1 <= N-1, TP <= n1, and TN <= N-n1
  int L = (TP > 1) ? TP : 1; // max(1, TP)
  int U1 = N - 1;
  int U2 = N - TN;
  int U = (U1 < U2) ? U1 : U2; // min(N-1, N-TN)
  
  if (L > U)
    reject("No feasible n1 given TP,TN,N (L>U).");
}

parameters {
  real<lower=0, upper=1> p1;
  real<lower=0, upper=1> p2;
  
  //negBin prior parameters for n1
  real<lower=0, upper=1> p3;//winBUGS dnegbin(p3, r): probability parameter
  real<lower=0> r; //shape param
  
}

model {
  //priors (same as winbugs)
  p1 ~ beta(a1, b1);
  p2 ~ beta(a2, b2);
  p3 ~ beta(a3, b3);
  r  ~ gamma(a_r, b_r);//stan gamma(shape, rate)
  
  //in winbugs n1 ~ dnegbin(p3, r) I(1, N-1)
  //normalize prior over {1,...,N-1}:
    // Zprior = P(1 <= n1 <= N-1 | p3, r) = CDF(N-1) - CDF(0)
  {
    real log_Zprior = log_diff_exp(neg_binomial_lcdf(N - 1 | r, p3),
                                   neg_binomial_lcdf(0 | r, p3));
    
    vector[U - L + 1] log_w;
    
    //sum only over feasible n1 (same posterior; impossible terms contribute 0 anyway)
    for (n1 in L:U) {
      int n2 = N - n1;
      
      log_w[n1 - L + 1] =
        binomial_lpmf(TP | n1, p1)
      + binomial_lpmf(TN | n2, p2)
      + neg_binomial_lpmf(n1 | r, p3)
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
  
  //draw n1 from p(n1 | p1, p2, p3, r, data) each iteration
  {
    real log_Zprior = log_diff_exp(neg_binomial_lcdf(N - 1 | r, p3),
                                   neg_binomial_lcdf(0 | r, p3));
    
    vector[U - L + 1] log_w;
    
    for (n1 in L:U) {
      int n2 = N - n1;
      
      log_w[n1 - L + 1] =
        binomial_lpmf(TP | n1, p1)
      + binomial_lpmf(TN | n2, p2)
      + neg_binomial_lpmf(n1 | r, p3)
      - log_Zprior;
    }
    
    n1_draw = (categorical_logit_rng(log_w) - 1) + L;
  }
  
  n2_draw = N - n1_draw;
  FP_draw = n1_draw - TP;
  FN_draw = n2_draw - TN;
}
