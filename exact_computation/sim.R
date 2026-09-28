source("core.R")
set.seed(20260927)
R <- 500
scen <- expand.grid(prev = c(0.1, 0.3, 0.5), Se = c(0.7, 0.9), Sp = c(0.7, 0.9), N = c(100, 500, 1000))
priors_p <- list(paper = c(a1 = 1, b1 = 0.1, a2 = 0.1, b2 = 1),     # as in manuscript joint model
                 flat  = c(a1 = 1, b1 = 1,   a2 = 1,   b2 = 1))     # Beta(1,1) on Se and FPR
qty <- c("n1","n2","FN","TN","Se","Sp","NPV","Acc")

out <- list(); k <- 0
for (N in unique(scen$N)) {
  lp1 <- c(-Inf, rep(0, N - 1))                                    # Model 1: uniform on 1..N-1
  lp2 <- m2_logprior(N, 1, 2 / N)                                  # Model 2: lambda ~ Gamma(1, rate 2/N), mean N/2
  gr3 <- m3_grid(N, 0.1, 0.5, 0.1, 0.01)                           # Model 3: manuscript hyperparameters
  cache3 <- new.env()
  for (s in which(scen$N == N)) {
    f_out <- sprintf("parts/scen_%02d.rds", s); if (file.exists(f_out)) next
    out <- list(); k <- 0; set.seed(20260927 + s)
    sc <- scen[s, ]; n1 <- round(N * sc$prev); n2 <- N - n1
    for (rep in 1:R) {
      TP <- rbinom(1, n1, sc$Se); FP <- rbinom(1, n2, 1 - sc$Sp)
      if (TP == 0) next                                             # n1 undefined support; skip (rare)
      FN <- n1 - TP; TN <- n2 - FP
      truth <- c(n1 = n1, n2 = n2, FN = FN, TN = TN, Se = TP / n1, Sp = TN / n2,
                 NPV = if (FN + TN > 0) TN / (FN + TN) else NA, Acc = (TP + TN) / N)
      key <- as.character(TP)
      if (is.null(cache3[[key]])) cache3[[key]] <- m3_logprior(gr3, TP, N)
      lps <- list(M1 = lp1, M2 = lp2, M3 = cache3[[key]])
      for (pp in names(priors_p)) for (m in names(lps)) {
        h <- priors_p[[pp]]
        po <- posterior_n1(TP, FP, N, lps[[m]], h["a1"], h["b1"], h["a2"], h["b2"])
        S <- summarise_fast(po, TP, FP, N); rownames(S) <- qty
        k <- k + 1
        out[[k]] <- data.frame(scen = s, N = N, prev = sc$prev, Se_true = sc$Se, Sp_true = sc$Sp,
          rep = rep, model = m, pprior = pp, qty = qty, truth = truth[qty],
          mean = S[qty, "mean"], med = S[qty, "med"], lo = S[qty, "lo"], hi = S[qty, "hi"],
          set_lo = po$lo, set_hi = po$hi, row.names = NULL)
      }
    }
    saveRDS(do.call(rbind, out), f_out); cat("done scenario", s, "\n")
  }
}
res <- do.call(rbind, lapply(sort(list.files("parts", full.names = TRUE)), readRDS))
saveRDS(res, "sim_raw.rds")
