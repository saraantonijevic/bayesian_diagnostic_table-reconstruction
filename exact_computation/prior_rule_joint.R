source("core.R")
TP <- 71; FP <- 28; N <- 182
bp <- function(mu, m) c(mu * m, (1 - mu) * m)
gr <- m3_grid(N, .1, .5, .1, .01, 100, 100)
lps <- list(M1 = c(-Inf, rep(0, N - 1)), M2 = m2_logprior(N, 1, 2 / N), M3 = m3_logprior(gr, TP, N))
for (m in c(5, 10, 20)) { s1 <- bp(.90, m); s2 <- bp(.28, m); cat("\nm =", m, "\n")
  for (k in names(lps)) { po <- posterior_n1(TP, FP, N, lps[[k]], s1[1], s1[2], s2[1], s2[2])
    fn <- po$n1 - TP; w <- po$w; q <- function(p) fn[which(cumsum(w) >= p)[1]]
    cat(k, ": FN mean", round(sum(w*fn), 1), " median", q(.5), " 95% (", q(.025), ",", q(.975), ")\n") } }
