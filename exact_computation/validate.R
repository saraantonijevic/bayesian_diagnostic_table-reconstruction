source("core.R")
TP <- 71; FP <- 28; N <- 182
for (G in c(40, 60, 100)) {
  gr <- m3_grid(N, 0.1, 0.5, 0.1, 0.01, G, G)
  lp<- m3_logprior(gr, TP, N)
  po<- posterior_n1(TP, FP, N, lp, 1, 0.1, 0.1, 1)
  cat("Model 3, paper priors, grid", G, ":\n"); print(round(summarise_post(po, TP, FP, N)[1:2, ], 3))
}
#posterior of p2 given the n2 posterior: mixture of Beta(FP + a2, n2 - FP + b2)
gr <- m3_grid(N, 0.1, 0.5, 0.1, 0.01, 60, 60); po <- posterior_n1(TP, FP, N, m3_logprior(gr, TP, N), 1, .1, .1, 1)
set.seed(1); n2s <- N - sample(po$n1, 1e5, TRUE, po$w); p2s <- rbeta(1e5, FP + 0.1, n2s - FP + 1)
cat("p2 posterior (FPR): mean", round(mean(p2s),3), " quantiles", round(quantile(p2s, c(.025,.5,.975)),3), "\n")
n1s <- N - n2s; p1s <- rbeta(1e5, TP + 1, n1s - TP + 0.1)
cat("p1 posterior (Se):  mean", round(mean(p1s),3), " quantiles", round(quantile(p1s, c(.025,.5,.975)),3), "\n")
cat("Identified set for n1: [", max(TP,1), ",", N - FP, "]\n")
