library(R2WinBUGS)

#WinBUGS path
options(bugs.directory = "C:\\Users\\saraa\\OneDrive\\Documents\\WinBugs\\winbugs14_full_patched\\WinBUGS14")
wd = "C:/WinBUGS_work"
if (!dir.exists(wd)) dir.create(wd, recursive = TRUE)

set.seed(123)

#observed data
TP = 71L #TP 
TN = 80L #TN

#priors for the p1 
a1 = 1
b1 = 1

#priors for the p2
a2 = 1
b2 = 1

#N prior support & weights
Nmin = TP + TN + 2L #ensures n1>=TP, n2>=TN, both >=1
Nmax = Nmin + 200L 
k_idx = 1:Nmax

#Poisson(lambdaN) truncated to Nmin,...,Nmax and renormalized
lambdaN = max(250, Nmin + 50) #sets Poisson mean for N (at least 250)
omegaRaw = dpois(k_idx, lambdaN) #computes Poisson pmf for each k
omegaRaw[k_idx < Nmin] = 0
omega = as.numeric(omegaRaw / sum(omegaRaw))
stopifnot(length(omega) == Nmax, isTRUE(all.equal(sum(omega), 1, tol=1e-12)))

#BUGS model
modfile = file.path(wd, "model_unknownN_binomial.txt")
writeLines(
  c(
    "model{",
    "  # Likelihood with dependent totals",
    "  TP ~ dbin(p1, n1)",
    "  n2 <- N - n1",
    "  TN ~ dbin(p2, n2)",
    "  FN <- n1 - TP",
    "  FP <- n2 - TN",
    "",
    "  # Priors for test characteristics",
    "  p1 ~ dbeta(a1, b1)",
    "  p2 ~ dbeta(a2, b2)",
    "",
    "  # Discrete prior for total N over 1..Nmax (weights omega[] from R)",
    "  N ~ dcat(omega[])",
    "",
    "  # Uniform prior for n1 over feasible {TP, ..., N - TN}",
    "  for(k in 1:Nmax){",
    "    in_lo[k] <- step(k - (TP - 0.5))",
    "    in_hi[k] <- step((N - TN + 0.5) - k)",
    "    inrange[k] <- in_lo[k] * in_hi[k]",
    "  }",
    "  support_len <- N - TN - TP + 1   # >= 1 because N >= TP+TN+2",
    "  invS <- 1 / support_len",
    "  for(k in 1:Nmax){",
    "    pi[k] <- inrange[k] * invS",
    "  }",
    "  n1 ~ dcat(pi[])",
    "}",
    ""
  ),
  con = modfile,
  useBytes = TRUE
)

dat = list( #refresh BUGS data + inits
  TP   = as.integer(TP),
  TN   = as.integer(TN),
  a1   = a1, b1 = b1,
  a2   = a2, b2 = b2,
  Nmax = as.integer(Nmax),
  omega = omega
)


initFunction = function() {  #function to have initial values for the MCMC
  N0 = max(Nmin, which.max(omega)) #feasible N start; start at N at the prior mode
  n1_min = TP  #smallest n1 is TP
  n1_max = N0 - TN #largest n1 ensures n2 = N0 - n1 >= TN
  n10 = max(n1_min, min(n1_max, floor((n1_min + n1_max)/2)))
  list(
    p1 = rbeta(1, a1, b1), #initializes sensitivity
    p2 = rbeta(1, a2, b2), #initializes specificity
    N  = as.integer(N0),
    n1 = as.integer(n10)
  )
}

pars = c("p1","p2","n1","n2","N","FN","FP")


fit = bugs(
  data = dat,
  inits = list(initFunction(), initFunction(), initFunction()),
  parameters.to.save = pars,
  model.file = modfile,
  n.chains = 3,
  n.iter = 3000,
  n.burnin = 500,
  n.thin = 1,
  DIC = FALSE,
  bugs.directory = getOption("bugs.directory"),
  working.directory = wd,
  debug = FALSE
)

#summary 
qs = function(z) c(mean = mean(z), sd = sd(z),
                    `2.5%` = unname(quantile(z, .025)),
                    `50%`  = median(z),
                    `97.5%`= unname(quantile(z, .975)))
post = fit$sims.list
tab = rbind(
  n1 = qs(post$n1),
  n2 = qs(post$n2),
  FN = qs(post$FN),
  FP = qs(post$FP),
  N  = qs(post$N)
)
print(round(as.data.frame(tab), 3))
