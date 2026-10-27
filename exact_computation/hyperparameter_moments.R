# Posterior mean and SD of the Model 3 hyperparameters p3 and r (Supplementary Tables S7 and S15). 
#Grid: equal-probability in p3, qually spaced in log r over (1e-200, 3000), so the heavy right tail of
# the Gamma(0.1, 0.01) prior on r is represented. Run from exact_computation/.
source("core.R")
hyper_moments <- function(TP, FP, N, Gp = 200, Gr = 4000, LOW = 1e-200, HIGH = 3000) {
  u <- (seq_len(Gp) - .5) / Gp
  lp3 <- log(qbeta(u, .1, .5)); l1 <- log(qbeta(1 - u, .5, .1)); p3 <- exp(lp3)
  lr <- seq(log(LOW), log(HIGH), length.out = Gr); r <- exp(lr)
  lw_r <- dgamma(r, .1, .01, log = TRUE) + lr # prior density x dr, in log-r coordinates
  n <- 0:(N - 1); n1 <- TP:(N - 1); cols <- n1 + 1
  ll <- lbb(TP, n1, 1, .1) + ifelse(N - n1 >= FP, suppressWarnings(lbb(FP, N - n1, .1, 1)), -Inf)
  Wrp <- matrix(0, Gr, Gp)
  for (j in seq_len(Gr)) {
    L <- outer(lp3, rep(1, N)) * r[j] + outer(l1, n) +
         matrix(lgamma(n + r[j]) - lgamma(r[j]) - lgamma(n + 1), Gp, N, byrow = TRUE)
    S <- L[, cols, drop = FALSE]
    S <- S - row_lse(S) # NegBin truncated to TP..N-1
    Wrp[j, ] <- row_lse(sweep(S, 2, ll, "+")) + lw_r[j] - log(Gp)
  }
  W <- exp(Wrp - lse(as.vector(Wrp))); wr <- rowSums(W); wp <- colSums(W)
  mr <- sum(wr * r); mp <- sum(wp * p3)
  c(p3_mean = mp, p3_sd = sqrt(sum(wp * (p3 - mp)^2)), r_mean = mr, r_sd = sqrt(sum(wr * (r - mr)^2)))
}
print(round(rbind(`Breast MRI (Table S15)`= hyper_moments(71, 28, 182),
                  `Wismueller (Table S7)`= hyper_moments(105, 17, 620)), 3))
