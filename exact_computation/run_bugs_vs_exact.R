ENGINE <- "winbugs"
BUGS_DIR <- "C:/Users/saraa/Downloads/winbugs14_full_patched/WinBUGS14"
source("incomplete2x2.R")

make_data <- function(TP, FP, N, Se = NA, Sp = NA, Acc = NA, Prev = NA, digits = 3,
                      hard = 1, tol = 0.005, alpha = c(1, 1, 1, 1)) {
  u <- function(x) as.numeric(!is.na(x)); v <- function(x) ifelse(is.na(x), 0, x)
  h <- 0.5 * 10^(-digits)
  list(TP = TP, FP = FP, N = N, alpha = alpha, ones = 1, hard = hard, tol = tol,
       useSe = u(Se), rSe = v(Se), hSe = h, useSp = u(Sp), rSp = v(Sp), hSp = h,
       useAcc = u(Acc), rAcc = v(Acc), hAcc = h, usePrev = u(Prev), rPrev = v(Prev), hPrev = h)
}

run_bugs <- function(dat, FN_init, n_iter = 50000, n_burn = 5000, n_chains = 3) {
  inits <- function() list(FN = FN_init, q1 = 0.5, q2 = 0.5, q3 = 0.5)
  if (ENGINE == "jags") {
    library(rjags)
    m <- jags.model("incomplete2x2_model.txt", dat, inits, n.chains = n_chains, quiet = TRUE)
    update(m, n_burn)
    s <- coda.samples(m, c("FN", "TN", "Se", "Sp", "Acc"), n_iter)
    list(draws = as.matrix(s), rhat = gelman.diag(s[, "FN"])$psrf[1])
  } else {
    library(R2WinBUGS)
    fit <- bugs(dat, inits, c("FN", "TN", "Se", "Sp", "Acc"), "incomplete2x2_model.txt",
                n.chains = n_chains, n.iter = n_iter + n_burn, n.burnin = n_burn,
                bugs.directory = BUGS_DIR, working.directory = getwd())
    list(draws = fit$sims.matrix, rhat = fit$summary["FN", "Rhat"])
  }
}

compare <- function(label, TP, FP, N, reported = list(), digits = 3, FN_init, ...) {
  ex <- complete_2x2(TP = TP, FP = FP, N = N, reported = reported, digits = digits)
  sx <- summary(ex)["FN", c("mean", "median", "lo95", "hi95")]
  dat <- do.call(make_data, c(list(TP = TP, FP = FP, N = N, digits = digits,
                 hard = as.numeric(ex$n_consistent > 0 || !length(reported))), reported))
  b <- run_bugs(dat, FN_init, ...)
  fn<- b$draws[,"FN"]
  sb <- c(mean = mean(fn), median = median(fn), lo95 = unname(quantile(fn, .025, type = 1)),
          hi95 = unname(quantile(fn, .975, type = 1)))
  cat(sprintf("\n%s\n", label))
  print(round(rbind(`exact (R)` = sx, `MCMC (BUGS)` = sb), 2))
  cat(sprintf("Rhat for FN: %.3f\n", b$rhat))
}

set.seed(1)
compare("Factory example: TP=48, FP=12, N=1000, accuracy 97% (2 digits)", 48, 12, 1000, list(Acc = .97), digits = 2, FN_init = 18)
compare("Breast MRI: TP=71, FP=28, N=182, sensitivity 96% (2 digits)",71, 28, 182, list(Se = .96), digits = 2, FN_init = 3)
compare("Wismueller: TP=105, FP=17, N=620, Se 95.0, Sp 96.7, Acc 96.4 (inconsistent -> soft)",105, 17, 620, list(Se = .950, Sp = .967, Acc = .964), digits = 3, FN_init = 6)
compare("Wismueller, counts only (no reported rates)",105, 17, 620, FN_init = 100)
