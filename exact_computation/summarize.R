res <- readRDS("sim_raw.rds")
res <- res[!is.na(res$truth) & !is.na(res$mean), ]
agg <- function(d) data.frame(
  bias = mean(d$mean - d$truth), rmse = sqrt(mean((d$mean - d$truth)^2)),
  cover = mean(d$lo <= d$truth & d$truth <= d$hi), width = mean(d$hi - d$lo),
  at_bound = mean(d$med == d$set_lo | d$med == d$set_hi), nrep = nrow(d))
keys <- c("N","prev","Se_true","Sp_true","model","pprior","qty")
sp <- split(res, res[keys], drop = TRUE)
tab <- do.call(rbind, lapply(sp, function(d) cbind(d[1, keys], agg(d))))
rownames(tab) <- NULL
tab$at_bound[tab$qty != "n1"] <- NA
write.csv(tab, "sim_summary.csv", row.names = FALSE)

#n1 coverage + Se/Sp RMSE averaged over scenarios
ov <- aggregate(cbind(cover, at_bound) ~ model + pprior, data = tab[tab$qty == "n1", ], FUN = mean)
rm <- aggregate(rmse ~ model + pprior + qty, data = tab[tab$qty %in% c("Se","Sp","NPV"), ], FUN = mean)
print(ov, digits = 3); print(reshape(rm, idvar = c("model","pprior"), timevar = "qty", direction = "wide"), digits = 3)
cat("\nn1 coverage by N x model x pprior:\n")
print(xtabs(cover ~ model + pprior + N, aggregate(cover ~ model + pprior + N, tab[tab$qty=="n1",], mean)), digits = 2)
cat("\nn1 coverage by prevalence x model (flat p-priors):\n")
print(xtabs(cover ~ model + prev, aggregate(cover ~ model + prev, tab[tab$qty=="n1" & tab$pprior=="flat",], mean)), digits = 2)
cat("\nn1 coverage by prevalence x model (paper p-priors):\n")
print(xtabs(cover ~ model + prev, aggregate(cover ~ model + prev, tab[tab$qty=="n1" & tab$pprior=="paper",], mean)), digits = 2)
cat("\nn1 coverage by Se x Sp x model (flat):\n")
print(ftable(xtabs(cover ~ model + Se_true + Sp_true, aggregate(cover ~ model + Se_true + Sp_true, tab[tab$qty=="n1" & tab$pprior=="flat",], mean))), digits = 2)
cat("\nn1 coverage by Se x Sp x model (paper):\n")
print(ftable(xtabs(cover ~ model + Se_true + Sp_true, aggregate(cover ~ model + Se_true + Sp_true, tab[tab$qty=="n1" & tab$pprior=="paper",], mean))), digits = 2)
cat("\nRelative bias of n1 (bias/n1) by prev x model x pprior:\n")
tab$relbias <- tab$bias / (tab$N * tab$prev)
print(ftable(xtabs(relbias ~ pprior + model + prev, aggregate(relbias ~ pprior + model + prev, tab[tab$qty=="n1",], mean))), digits = 2)
cat("\nIdentified-set width / N (mean):", round(mean((res$set_hi - res$set_lo)[res$qty=="n1"] / res$N[res$qty=="n1"]), 3), "\n")
