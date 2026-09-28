
#Give whatever is known: any cells, N, row/column totals, and any reported rates (possibly rounded). The function enumerates every
# table consistent with the hard information, weights each by a prior and by how well it reproduces the reported rates, and returns the
#posterior over tables plus summaries of cells and derived measures.
.measures <- function(T) {
  with(T, {
    sdiv <- function(a, b) ifelse(b > 0, a / b, NA_real_)
    data.frame(Se = sdiv(TP, TP + FN), Sp = sdiv(TN, TN + FP), PPV = sdiv(TP, TP + FP),
               NPV = sdiv(TN, TN + FN), Acc = sdiv(TP + TN, N), Prev = sdiv(TP + FN, N),
               F1 = sdiv(2 * TP, 2 * TP + FP + FN))
  })
}
.lse <- function(x) { m <- max(x); m + log(sum(exp(x - m))) }

## Dirichlet-multinomial log pmf of a table given its N (alpha named TP,FP,FN,TN)
.ldm <- function(T, alpha) {
  A <- sum(alpha); x <- as.matrix(T[, c("TP","FP","FN","TN")])
  lgamma(T$N + 1) + lgamma(A) - lgamma(T$N + A) +
    rowSums(lgamma(sweep(x, 2, alpha, "+")) - lgamma(x + 1)) - sum(lgamma(alpha))
}

complete_2x2 <- function(TP = NA, FP = NA, FN = NA, TN = NA, N = NA,
 pred_pos = NA, pred_neg = NA, act_pos = NA, act_neg = NA,
        reported = list(), digits = 3, tol = NULL,
       prior = c(TP = 1, FP = 1, FN = 1, TN = 1),
    N_max = NA, N_prior = c("loguniform", "uniform"),
    max_tables = 5e6) {
  N_prior <- if (is.function(N_prior)) N_prior else match.arg(N_prior)
  cells <- c(TP = TP, FP = FP, FN = FN, TN = TN)
  known <- !is.na(cells); unk <- names(cells)[!known]; s_known <- sum(cells[known])

  #enumerate candidate tables
  if (!is.na(N)) {
    R <- N - s_known; if (R < 0) stop("Known cells exceed N.")
    k <- length(unk)
    if (k == 0) G <- data.frame(row.names = 1) else {
      if ((R + 1)^(k - 1) > max_tables) stop("Too many candidate tables; add information.")
      G <- if (k == 1) data.frame(R) else expand.grid(rep(list(0:R), k - 1))
      if (k > 1) { G <- G[rowSums(G) <= R, , drop = FALSE]; G$last <- R - rowSums(G) }
      names(G) <- unk
    }
  } else {
    if (is.na(N_max)) stop("N is unknown: supply N_max, a plausible upper bound on the total.")
    U <- N_max - s_known; k <- length(unk)
    if ((U + 1)^k > max_tables) stop("Too many candidate tables; lower N_max or add information.")
    G <- expand.grid(rep(list(0:U), k)); names(G) <- unk
    G <- G[rowSums(G) <= U, , drop = FALSE]
  }
  T <- data.frame(TP = if (known["TP"]) TP else G$TP, FP = if (known["FP"]) FP else G$FP,
                  FN = if (known["FN"]) FN else G$FN, TN = if (known["TN"]) TN else G$TN)
  T$N <- T$TP + T$FP + T$FN + T$TN

  #hard constraints from margins
  keep <- T$N > 0
  if (!is.na(pred_pos)) keep <- keep & T$TP + T$FP == pred_pos
  if (!is.na(pred_neg))keep <- keep & T$FN + T$TN == pred_neg
  if (!is.na(act_pos)) keep <- keep & T$TP + T$FN == act_pos
  if (!is.na(act_neg))keep <- keep & T$FP + T$TN == act_neg
  T <- T[keep, , drop = FALSE]
  if (nrow(T) == 0) stop("No table satisfies the given counts and totals.")
  M <- .measures(T)

  #prior over tables
  lp <- if (is.function(prior)) prior(T) else .ldm(T, prior[c("TP","FP","FN","TN")])
  if (is.na(N)) lp <- lp + (if (is.function(N_prior)) N_prior(T$N) else
                            if (N_prior == "loguniform") -log(T$N) else 0)

  ## reported rates: rounding-consistent tables get full weight, others are penalised smoothly 
  ## if some tables reproduce every reported rate after rounding, keep only those (hard constraint); if none do, fall back to a soft penalty, sd 0.005.
  consistent <- rep(TRUE, nrow(T)); gaps <- list()
  for (nm in names(reported)) {
    r <- reported[[nm]]; d <- if (length(r) > 1) r[2] else digits; r <- r[1]
    h <- 0.5 * 10^(-d); x <- M[[nm]]
    if (is.null(x)) stop("Unknown measure: ", nm, ". Use Se, Sp, PPV, NPV, Acc, Prev, F1.")
    gap <- pmax(0, abs(x - r) - h - 1e-9); gap[is.na(gap)] <- Inf  # rounding boundary counted as consistent
    consistent <- consistent & gap == 0; gaps[[nm]] <- gap
  }
  if (length(reported)) {
    if (is.null(tol)) {
      if (any(consistent)) lp[!consistent] <- -Inf else tol <- 0.005
    }
    if (!is.null(tol)) for (g in gaps) lp <- lp - 0.5 * (g / tol)^2
  }
  full_range <- sapply(c("TP","FP","FN","TN","N","Se","Sp","PPV","NPV","Acc","Prev"), function(v) {
    x <- c(T[[v]], M[[v]]); x <- if (v %in% names(T)) T[[v]] else M[[v]]
    c(min = suppressWarnings(min(x, na.rm = TRUE)), max = suppressWarnings(max(x, na.rm = TRUE))) })
  n_cand <- nrow(T)
  ok <- is.finite(lp); T <- T[ok, ]; M <- M[ok, ]; lp <- lp[ok]; consistent <- consistent[ok]
  w <- exp(lp - .lse(lp))

  out <- list(tables = cbind(T, M, post = w, rounding_consistent = consistent), n_candidates = n_cand, full_range = full_range, n_consistent = sum(consistent), reported = reported)
  class(out) <- "complete2x2"; out
}

.wq <- function(x, w, p) { o <- order(x); cw <- cumsum(w[o]); sapply(p, function(pp) x[o][which(cw >= pp - 1e-12)[1]]) }

summary.complete2x2 <- function(object, ...) {
  D <- object$tables; w <- D$post
  vars <- c("TP","FP","FN","TN","N","Se","Sp","PPV","NPV","Acc","Prev")
  t(sapply(vars, function(v) { x <- D[[v]]; ok <- !is.na(x); ww <- w[ok] / sum(w[ok]); x <- x[ok]
    c(mean = sum(ww * x), median = .wq(x, ww, .5), lo95 = .wq(x, ww, .025), hi95 = .wq(x, ww, .975),
      min_possible = object$full_range["min", v], max_possible = object$full_range["max", v]) }))
}

print.complete2x2 <- function(x, digits = 3, top = 5, ...) {
  D <- x$tables
  cat("Candidate tables:", x$n_candidates)
  if (length(x$reported)) cat("   |  exactly consistent with reported rates:", x$n_consistent,
      if (x$n_consistent == 0) " (none: reported rates are mutually inconsistent; using nearest tables)" else "")
  cat("\n\nPosterior summary (min/max_possible = full range allowed by the given counts):\n")
  print(format(as.data.frame(round(summary(x), digits)), scientific = FALSE, drop0trailing = TRUE))
  cat("\nMost probable tables:\n")
  o <- head(order(-D$post), top)
  print(transform(D[o, c("TP","FP","FN","TN","N","Se","Sp","Acc")], post = signif(D$post[o], 3),
                  Se = round(Se, 4), Sp = round(Sp, 4), Acc = round(Acc, 4)), row.names = FALSE)
  invisible(x)
}


#for learning the prior from COMPLETE tables in the same field (Dirichlet-multinomial maximum likelihood)
fit_prior_from_tables <- function(tabs) {
  tabs <- as.data.frame(tabs)[, c("TP","FP","FN","TN")]; tabs$N <- rowSums(tabs)
  #alpha = s * pi, pi on the simplex 
  to_alpha <- function(th) { e <- exp(c(th[1:3], 0)); exp(th[4]) * e / sum(e) }
  nll <- function(th) -sum(.ldm(tabs, setNames(to_alpha(th), c("TP","FP","FN","TN"))))
  m <- colMeans(tabs[, 1:4] + 0.5); init <- c(log(m[1:3] / m[4]), log(10))
  fit <- optim(init, nll, method = "L-BFGS-B", lower = c(-20, -20, -20, log(0.1)), upper = c(20, 20, 20, log(1e4)))
  setNames(to_alpha(fit$par), c("TP","FP","FN","TN"))
}
