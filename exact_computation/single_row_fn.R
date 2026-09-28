#Exact posterior for the single-row model (normalized truncation, r >= 1)
#y|n,p ~ Bin(n,p); p ~ Beta(al,be); n|p*,r ~ NegBin(p*,r) truncated to y <= n (<= Nup)
#r|lam ~ Pois(lam), r >= 1
#lam ~ Gamma(a, b)
#p* ~ Beta(as, bs)
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
      ld - lz }) + lbbn + lPr[i] - log(Gp)# length(n) x Gp
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
  out<- rbind(lambda = sm(lam), n = ws(n, wn), p = sm(psamp), pstar = ws(ps, wp), r = ws(r, wr))
  colnames(out) <- c("Mean","SD","2.5%","Median","97.5%")
  attr(out, "tail") <- if (is.na(Nup)) sum(wn[n > 0.9 * cap]) else 0
  out
}
