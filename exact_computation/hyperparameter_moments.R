# Posterior summaries (mean, SD, 2.5%, median, 97.5%) of the Model 3 hyperparameters
# p3 and r (Supplementary Tables S7 and S15).
# Grid: equal-probability (prior-CDF) in p3, equally spaced in log r over (1e-200, 3000),
# so the heavy right tail of the Gamma(0.1, 0.01) prior on r is represented.
# Quantiles interpolate the cumulative posterior within grid cells.
# Run from exact_computation/. Takes a few minutes.
source("core.R")
hyper_summary <- function(TP, FP, N, Gp = 1000, Gr = 1500, LOW = 1e-200, HIGH = 3000) {
  u <- (seq_len(Gp) - .5) / Gp
  p3 <- qbeta(u, .1, .5); lp3 <- log(p3); l1 <- log(qbeta(1 - u, .5, .1))
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
  # quantiles: each node carries the mass of its cell [t - h/2, t + h/2] in the grid coordinate t
  q <- function(t, w, back) { h <- t[2] - t[1]; cw <- c(0, cumsum(w))
    sapply(c(.025, .5, .975), function(a) { i <- which(cw[-1] >= a)[1]; back(t[i] - h/2 + (a - cw[i]) / w[i] * h) }) }
  mp <- sum(wp * p3); mr <- sum(wr * r)
  out <- rbind(p3 = c(mp, sqrt(sum(wp * (p3 - mp)^2)), q(u, wp, function(x) qbeta(x, .1, .5))),
               r  = c(mr, sqrt(sum(wr * (r - mr)^2)), q(lr, wr, exp)))
  colnames(out) <- c("Mean", "SD", "2.5%", "Median", "97.5%"); out
}
cat("Breast MRI (Table S15)\n"); print(signif(hyper_summary(71, 28, 182), 4))
cat("\nWismueller (Table S7)\n");  print(signif(hyper_summary(105, 17, 620), 4))
