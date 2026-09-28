## Simulation: same 36 scenarios as sim.R, but the report also gives sensitivity rounded to a whole percent
source("core.R")
R <- 500
scen <- expand.grid(prev = c(0.1, 0.3, 0.5), Se = c(0.7, 0.9), Sp = c(0.7, 0.9), N = c(100, 500, 1000))
priors <- list(paper = c(1, .1, .1, 1), flat = c(1, 1, 1, 1))
rows <- list(); k <- 0
for (N in unique(scen$N)) {
  lps0 <- list(M1 = c(-Inf, rep(0, N - 1)), M2 = m2_logprior(N, 1, 2 / N)); gr3 <- m3_grid(N, .1, .5, .1, .01); cache <- new.env()
  for (s in which(scen$N == N)) {
    set.seed(20260927 + 100 + s); sc <- scen[s, ]; n1 <- round(N * sc$prev); n2 <- N - n1
    for (r in 1:R) {
      TP <- rbinom(1, n1, sc$Se); FP <- rbinom(1, n2, 1 - sc$Sp); if (TP == 0) next
      FN <- n1 - TP; rSe <- round(TP / n1, 2)
      key <- as.character(TP); if (is.null(cache[[key]])) cache[[key]] <- m3_logprior(gr3, TP, N)
      lps <- c(lps0, list(M3 = cache[[key]]))
      for (pp in names(priors)) for (m in names(lps)) {
        h <- priors[[pp]]; po <- posterior_n1(TP, FP, N, lps[[m]], h[1], h[2], h[3], h[4])
        fn <- po$n1 - TP; ok <- abs(TP / po$n1 - rSe) <= 0.005 + 1e-9
        lw <- log(po$w) + ifelse(ok, 0, -Inf); w <- exp(lw - lse(lw))
        lo <- fn[which(cumsum(w) >= .025)[1]]; hi <- fn[which(cumsum(w) >= .975)[1]]
        k <- k + 1; rows[[k]] <- data.frame(s = s, N = N, model = m, pprior = pp, cover = as.numeric(lo <= FN & FN <= hi),
                                     width = hi - lo, err = sum(w * fn) - FN, setw = max(fn) - min(fn))
      }
    }
    cat("done", s, "\n")
  }
}
d <- do.call(rbind, rows)
saveRDS(d, "sim_reported_raw.rds")
a <- aggregate(cbind(cover, width, rmse = err^2) ~ model + pprior, d, mean); a$rmse <- sqrt(a$rmse); print(a, digits = 3)
b <- aggregate(cbind(cover, width) ~ model + pprior + N, d, mean); print(b, digits = 3)
