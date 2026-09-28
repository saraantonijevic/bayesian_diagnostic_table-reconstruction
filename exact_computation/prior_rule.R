source("single_row_fn.R")
Se0 <- 0.90; FPR0 <- 0.28
bp <- function(mu, m) c(mu * m, (1 - mu) * m)
settings <- list(`rule m=10` = 10, `m=5` = 5, `m=20` = 20)
out <- list()
for (nm in names(settings)) { m <- settings[[nm]]; s1 <- bp(Se0, m); s2 <- bp(FPR0, m)
  d<- single_row_full(71, 1, .1, s1[1], s1[2], 1, 1, cap = 5000, Gp = 150)["n", ]
  dN <- single_row_full(71, 1, .1, s1[1], s1[2], 1, 1, Nup = 182, Gp = 150)["n", ]
  nd <- single_row_full(28, 1, .1, s2[1], s2[2], 1, 1, cap = 20000, Gp = 150)
  ndN<- single_row_full(28, 1, .1, s2[1], s2[2], 1, 1, Nup = 182, Gp = 150)["n", ]
  out[[nm]] <- rbind(dis = d, dis_N = dN, non = nd["n", ], non_N = ndN)
  cat("\n", nm, " Se prior Beta(", s1, ")  FPR prior Beta(", s2, ")  tail(non):", signif(attr(nd, "tail"), 2), "\n")
  print(round(out[[nm]], 1)) }
# flat p priors for comparison
d <- single_row_full(71, 1, .1, 1, 1, 1, 1, cap = 5000, Gp = 150)["n", ]
nd <- single_row_full(28, 1, .1, 1, 1, 1, 1, cap = 20000, Gp = 150)
cat("\n flat Beta(1,1) on p, tail(non):", signif(attr(nd,"tail"),2), "\n")
print(round(rbind(dis = d, non = nd["n", ]), 1))

