#Exact (non-MCMC) posterior for n1 in the known-N models (Models 1-3)
#p1, p2 are integrated out analytically -> beta-binomial marginal likelihoods. Hyperparameters of the n1 prior (lambda; p3, r) are integrated numerically on
# equal-probability quantile grids of their priors.

lse <- function(x) { m <- max(x); if (!is.finite(m)) return(m); m + log(sum(exp(x - m))) }
row_lse <- function(M) { m <- M[cbind(seq_len(nrow(M)), max.col(M, ties.method = "first"))]
  m + log(rowSums(exp(M - m))) }
lbb <- function(y, n, a, b) lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)

# Model 2: n1 | lambda ~ Pois(lambda) truncated to {1..N-1}, lambda ~ Gamma(al, bl) (rate)
m2_logprior <- function(N, al, bl, G = 400) {
  lam <- qgamma((seq_len(G) - 0.5) / G, al, bl)
  n <- 0:(N - 1)
  L <- outer(lam, n, function(l, k) dpois(k, l, log = TRUE))# G x N
  logZ <- row_lse(L[, -1, drop = FALSE])# support 1..N-1
  lp <- apply(L - logZ, 2, lse) - log(G)# log prior on n=0..N-1
  lp[1] <- -Inf
  lp
}

#Model 3: n1 | p3, r ~ NegBin(p3, r) truncated to {TP..N-1}, p3 ~ Beta(a3,b3), r ~ Gamma(ar,br)
m3_grid <- function(N, a3, b3, ar, br, Gp = 60, Gr = 60) {
  u <- (seq_len(Gp) - 0.5) / Gp
  lp3 <- log(qbeta(u, a3, b3)); l1mp3 <- log(qbeta(1 - u, b3, a3))# log p3, log(1-p3), stable
  r<- qgamma((seq_len(Gr) - 0.5) / Gr, ar, br)
  g <- expand.grid(i = seq_len(Gp), j = seq_len(Gr))
  n<- 0:(N - 1)
  rr <- r[g$j]
  L <- lgamma(outer(rr, n, "+")) - lgamma(rr) -
        matrix(lgamma(n + 1), nrow(g), N, byrow = TRUE) +
        rr * lp3[g$i] + outer(l1mp3[g$i], n) # log NB pmf, G x N
  list(L = L, G = nrow(g))
}
m3_logprior <- function(grid, TP, N) {
  cols <- (TP + 1):N # n = TP..N-1
  S <- grid$L[, cols, drop = FALSE]
  logZ <- row_lse(S)
  lp <- rep(-Inf, N); lp[cols] <- apply(S - logZ, 2, lse) - log(grid$G)
  lp
}

#posterior over n1 and derived table quantities
posterior_n1 <- function(TP, FP, N, logprior, a1, b1, a2, b2) {
  lo <- max(TP, 1)
  hi<- min(N - FP, N - 1)
  n1<- lo:hi
  lpost <- logprior[n1 + 1] + lbb(TP, n1, a1, b1) + lbb(FP, N - n1, a2, b2)
  w <- exp(lpost - lse(lpost))
  list(n1 = n1, w = w, lo = lo, hi = hi)
}
wq <- function(x, w, p) { o <- order(x)
cw <- cumsum(w[o]); sapply(p, function(pp) x[o][which(cw >= pp - 1e-12)[1]]) }

summarise_post <- function(post, TP, FP, N) {
  n1 <- post$n1; w <- post$w; n2 <- N - n1
  q <- list(n1 = n1, n2 = n2, FN = n1 - TP, TN = n2 - FP,
            Se = TP / n1, Sp = (n2 - FP) / n2,
            NPV = if (N - TP - FP > 0) (n2 - FP) / (N - TP - FP) else rep(NA, length(n1)),
            Acc = (TP + n2 - FP) / N)
  t(sapply(q, function(x) if (anyNA(x)) c(mean = NA, med = NA, lo = NA, hi = NA) else
    c(mean = sum(w * x), med = wq(x, w, .5), lo = wq(x, w, .025), hi = wq(x, w, .975))))
}

# every table quantity is monotone in n1, so quantiles come from the n1 CDF
summarise_fast <- function(post, TP, FP, N) {
  n1 <- post$n1; w <- post$w; n2 <- N - n1
  cw <- cumsum(w); rcw <- rev(cumsum(rev(w)))
  qi_inc <- function(p) which(cw  >= p - 1e-12)[1]# for increasing f(n1)
  qi_dec <- function(p) max(which(rcw >= p - 1e-12)) # for decreasing f(n1)
  nneg <- N - TP - FP
  X <- cbind(n1 = n1, n2 = n2, FN = n1 - TP, TN = n2 - FP, Se = TP / n1, Sp = (n2 - FP) / n2,
             NPV = if (nneg > 0) (n2 - FP) / nneg else NA, Acc = (TP + n2 - FP) / N)
  inc <- c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE)
  t(sapply(seq_len(ncol(X)), function(j) {
    x <- X[, j]; f <- if (inc[j]) qi_inc else qi_dec
    c(mean = sum(w * x), med = x[f(.5)], lo = x[f(.025)], hi = x[f(.975)]) }))
}
