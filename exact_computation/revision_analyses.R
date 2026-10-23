#analysis for revisions
source("core.R")

#FN interval from a posterior over n1 (FN = n1 - TP is increasing in n1)
fn_summary <- function(n1, w, TP) {
  fn <- n1 - TP; cw <- cumsum(w)
  c(mean = sum(w * fn), lo = fn[which(cw >= .025 - 1e-12)[1]], hi = fn[which(cw >= .975 - 1e-12)[1]])
}
bp <- function(mu, m) c(mu * m, (1 - mu) * m)#Beta(m*mu, m*(1-mu))

#breast-MRI benchmark
TP <- 71
FP <- 28
N <- 182
Se0 <- 0.90
Sp0 <- 0.72
lp_unif <- c(-Inf, rep(0, N - 1))#model 1: uniform on 1..N-1

cat("accuracy-prior concentration m (joint fixed-N, model 1); true FN = 3\n")
for (m in c(2, 5, 10, 20, 50)) {
  s1 <- bp(Se0, m); s2 <- bp(1 - Sp0, m)
  po <- posterior_n1(TP, FP, N, lp_unif, s1[1], s1[2], s2[1], s2[2])
  r <- fn_summary(po$n1, po$w, TP)
  cat(sprintf("m = %2d: mean FN %.2f  95%% (%d, %d)\n", m, r["mean"], r["lo"], r["hi"]))
}

cat("\ncrossed accuracy x denominator priors (joint fixed-N); true FN = 3\n")
# beta-binomial prevalence prior: n1 | q ~ Bin(N, q), q ~ Beta(a, b); log prior on n = 0..N-1
lp_bb <- function(a, b) { lp <- lbb(0:(N - 1), N, a, b); lp[1] <- -Inf; lp }
den <- list("Discrete uniform" = lp_unif, "Beta-binomial (2,2)" = lp_bb(2, 2), "Beta-binomial (1,2)" = lp_bb(1, 2))
acc <- list("Uniform accuracy" = c(1, 1, 1, 1),
            "External, m = 10" = c(bp(Se0, 10), bp(1 - Sp0, 10)))
for (d in names(den)) {
  cells <- sapply(acc, function(h) {
    po <- posterior_n1(TP, FP, N, den[[d]], h[1], h[2], h[3], h[4])
    r <- fn_summary(po$n1, po$w, TP); sprintf("%.1f (%d, %d)", r["mean"], r["lo"], r["hi"])
  })
  cat(sprintf("%-20s %-16s %s\n", d, cells[1], cells[2]))
}

#Wismueller and Stockmaster: tau sensitivity 
cat("\nsoft-weight tolerance tau (Wismueller & Stockmaster, Model 1)\n")
TPw <- 105
FPw <- 17
Nw <- 620
rep_val <- c(Se = 0.950, Sp = 0.967, Acc = 0.964); h <- 0.0005 # one decimal in percent
po0 <- posterior_n1(TPw, FPw, Nw, c(-Inf, rep(0, Nw - 1)), 1, 0.1, 0.1, 1)
n2 <- Nw - po0$n1
g <- cbind(Se = TPw / po0$n1, Sp = (n2 - FPw) / n2, Acc = (TPw + n2 - FPw) / Nw)
delta <- abs(sweep(g, 2, rep_val)) - h; delta[delta < 0] <- 0
for (tau in c(0.001, 0.0025, 0.005, 0.01, 0.02)) {
  lw <- log(po0$w) - 0.5 * rowSums(delta^2) / tau^2
  r <- fn_summary(po0$n1, exp(lw - lse(lw)), TPw)
  cat(sprintf("tau = %.4f: mean FN %.2f  95%% (%d, %d)\n", tau, r["mean"], r["lo"], r["hi"]))
}

#per-model coverage from the main simulation 
cat("\nPer-model coverage of FN, counts only\n")
f_raw <- if (file.exists("sim_raw.rds")) "sim_raw.rds" else "../../sim_raw.rds"
if (file.exists(f_raw)) {
  res <- readRDS(f_raw); d <- res[res$qty == "FN", ]
  d$cover <- d$lo <= d$truth & d$truth <= d$hi
  d$width <- d$hi - d$lo
  a <- aggregate(cbind(cover, width) ~ pprior + model, d, mean)
  a$cover <- round(100 * a$cover, 1); a$width <- round(a$width, 1)
  print(a, row.names = FALSE)
  cat(sprintf("Uniform priors averaged over models: %.1f%%, width %.1f (%d tables)\n",
              100 * mean(d$cover[d$pprior == "flat"]), mean(d$width[d$pprior == "flat"]),
              nrow(d) / 6))
} else cat("sim_raw.rds not found: run sim.R first\n")

#reporting-discrepancy experiment 
cat("\n perturbed reported sensitivity (N = 500, prev 0.30, Se 0.85, Sp 0.90; Model 1, uniform priors)\n")
set.seed(271828)
R <- 1000
N <- 500
hs <- 0.005 # whole-percent rounding
taus <- c(0.005, 0.01, 0.02)
out <- matrix(NA, R, 2 + 2 * length(taus))
for (i in 1:R) {
  n1 <- rbinom(1, N, 0.30); TP <- rbinom(1, n1, 0.85); FP <- rbinom(1, N - n1, 0.10)
  FN <- n1 - TP; se <- TP / n1
  r_exact <- round(se, 2)
  r_pert<- round(min(1, max(0, se + rnorm(1, 0, 0.01))), 2)
  po <- posterior_n1(TP, FP, N, c(-Inf, rep(0, N - 1)), 1, 1, 1, 1)
  g <- TP / po$n1
  ok <- abs(g - r_exact) <= hs + 1e-9
  w <- ifelse(ok, po$w, 0); s <- fn_summary(po$n1, w / sum(w), TP)
  res_i <- c(s["lo"] <= FN & FN <= s["hi"], s["hi"] - s["lo"])
  for (tau in taus) {
    lw <- log(po$w) - 0.5 * pmax(0, abs(g - r_pert) - hs)^2 / tau^2
    s <- fn_summary(po$n1, exp(lw - lse(lw)), TP)
    res_i <- c(res_i, s["lo"] <= FN & FN <= s["hi"], s["hi"] - s["lo"])
  }
  out[i, ] <- res_i
}
lab <- c("Exact rounding, hard constraint", sprintf("Perturbed, tau = %.3f", taus))
for (j in seq_along(lab)) {
  cv <- mean(out[, 2 * j - 1])
  cat(sprintf("%-32s coverage %5.1f%% (MC SE %.1f)  mean width %.1f\n", lab[j],
              100 * cv, 100 * sqrt(cv * (1 - cv) / R), mean(out[, 2 * j])))
}
