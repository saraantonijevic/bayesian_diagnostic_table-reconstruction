#Exact posterior for the single-row model

#y|n,p ~ Bin(n,p); p ~ Beta(al,be); n|p*,r ~ NegBin(p*,r) truncated to y <= n

#r|lam ~ Pois(lam), r >= 1; lam ~ Gamma(a, b); p* ~ Beta(as, bs)
source("core.R")
single_row_full <- function(y, a, b, al, be, as, bs, Nup = NA, cap = 20000, Gp = 300, nsamp = 4e5) {
  hi <- if (is.na(Nup)) cap else Nup; n <- y:hi
  rmax <- qnbinom(1 - 1e-9, size = a, prob = b / (1 + b)); r <- 1:rmax
  lPr <- dnbinom(r, size = a, prob = b / (1 + b), log = TRUE)
  ps <- qbeta((1:Gp - .5) / Gp, as, bs)
  lbbn <- lbb(y, n, al, be)
  Wn <- rep(-Inf, length(n)); Wr <- rep(-Inf, length(r)); Wp <- matrix(-Inf, length(r), Gp)
  for (i in seq_along(r)) {
    M <- sapply(ps, function(p) { ld <- dnbinom(n, r[i], p, log = TRUE)
      lz <- if (is.na(Nup)) pnbinom(y - 1, r[i], p, lower.tail = FALSE, log.p = TRUE) else lse(ld)
      ld - lz }) + lbbn + lPr[i] - log(Gp) # length(n) x Gp
    M[!is.finite(M)] <- -Inf
    Wn <- apply(cbind(Wn, apply(M, 1, lse)), 1, lse)
    Wp[i, ] <- apply(M, 2, lse); Wr[i] <- lse(Wp[i, ])
  }
  tot <- lse(Wr); wn <- exp(Wn - tot); wr <- exp(Wr - tot); wp <- colSums(exp(Wp - tot))
  wq<- function(x, w) { o <- order(x); cw <- cumsum(w[o]); x[o][sapply(c(.025,.5,.975), function(q) which(cw >= q)[1])] }
  ws <- function(x, w) c(mean = sum(w*x), sd = sqrt(sum(w*x^2) - sum(w*x)^2), wq(x, w))
  set.seed(1)
  ns <- sample(n, nsamp, TRUE, wn); psamp <- rbeta(nsamp, y + al, ns - y + be)
  rs<- sample(r, nsamp, TRUE, wr); lam <- rgamma(nsamp, a + rs, b + 1)
  sm<- function(x) c(mean = mean(x), sd = sd(x), quantile(x, c(.025, .5, .975)))
  out <- rbind(lambda = sm(lam), n = ws(n, wn), p = sm(psamp), pstar = ws(ps, wp), r = ws(r, wr))
  colnames(out) <- c("Mean","SD","2.5%","Median","97.5%")
  attr(out, "tail") <- if (is.na(Nup)) sum(wn[n > 0.9 * cap]) else 0
  out
}
cases <- list(
  "Table 5: MRI diseased, single row"= list(y=71, a=1,b=.1, al=2,be=1, as=1,bs=1, cap=5000),"Table 6: MRI non-diseased, single row" = list(y=28, a=2,b=1,  al=2,be=5, as=1,bs=50, cap=40000), "Table 9: MRI diseased, known N" = list(y=71, a=1,b=.1, al=2,be=1, as=1,bs=1, Nup=182),
  "Table 10: MRI non-diseased, known N" = list(y=28, a=2,b=1,  al=2,be=5, as=1,bs=50, Nup=182), "Table S2: Svirsky diseased"= list(y=93, a=1,b=.1, al=2,be=1, as=1,bs=1, cap=5000), "Table S3: Svirsky non-diseased" = list(y=150,a=2,b=1,  al=2,be=5, as=1,bs=50, cap=60000))
res <- list()
for (nm in names(cases)) { x <- res[[nm]] <- do.call(single_row_full, cases[[nm]])
  cat(nm, "(posterior mass in top 10% of n range:", signif(attr(x, "tail"), 2), ")\n"); print(signif(x, 4)) }
