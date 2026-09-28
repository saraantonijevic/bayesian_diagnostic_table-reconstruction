source("core.R")
source("incomplete2x2.R")
#posterior over n1 under Models 1-3 (manuscript priors) times rounding indicator for reported rates
post_with_reported <- function(TP, FP, N, logprior, h, rep) {
  po <- posterior_n1(TP, FP, N, logprior, h[1], h[2], h[3], h[4])
  n1 <- po$n1; n2 <- N - n1; TN <- n2 - FP
  M <- list(Se = TP/n1, Sp = TN/n2, Acc = (TP+TN)/N)
  lw <- log(po$w)
  for (nm in names(rep)) { r <- rep[[nm]]; hw <- 0.5*10^(-r[2]); gap <- pmax(0, abs(M[[nm]]-r[1]) - hw - 1e-9)
    lw <- lw + if (all(gap > 0)) -0.5*(gap/0.005)^2 else ifelse(gap > 0, -Inf, 0) }
  w <- exp(lw - lse(lw)); fn <- n1 - TP
  c(mean = sum(w*fn), lo = fn[which(cumsum(w) >= .025)[1]], hi = fn[which(cumsum(w) >= .975)[1]])
}
paper <- c(1, .1, .1, 1)
ex <- list(Wis = list(TP=105, FP=17, N=620, rep=list(Se=c(.950,3), Sp=c(.967,3), Acc=c(.964,3))),
           MRI = list(TP=71, FP=28, N=182, rep=list(Se=c(.96,2))))
for (e in names(ex)) { x <- ex[[e]]; N <- x$N
  gr <- m3_grid(N, .1, .5, .1, .01)
  lps <- list(M1 = c(-Inf, rep(0, N-1)), M2 = m2_logprior(N, 1, 2/N), M3 = m3_logprior(gr, x$TP, N))
  cat("\n", e, "\n")
  for (m in names(lps)) { cat(m, "counts only :", round(post_with_reported(x$TP, x$FP, N, lps[[m]], paper, list()), 2),
                              "| + reported:", round(post_with_reported(x$TP, x$FP, N, lps[[m]], paper, x$rep), 2), "\n") }
  d <- complete_2x2(TP=x$TP, FP=x$FP, N=N, reported=lapply(x$rep, identity), digits=3)
  cat("DM uniform + reported:", round(summary(d)["FN", c(1,3,4)], 2), "\n")
}
