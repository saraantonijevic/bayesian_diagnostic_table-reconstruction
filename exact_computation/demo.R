source("incomplete2x2.R")

cat("1. Wismueller: only TP, FP, N ")
print(complete_2x2(TP = 105, FP = 17, N = 620))

cat("1b. Wismueller: + reported Se 95.0%, Sp 96.7%, Acc 96.4% ")
print(complete_2x2(TP = 105, FP = 17, N = 620, reported = list(Se = .950, Sp = .967, Acc = .964)))

cat("\n 2. Breast MRI benchmark (truth FN = 3, TN = 80): TP, FP, N + a rounded Se of 96%\n")
print(complete_2x2(TP = 71, FP = 28, N = 182, reported = list(Se = c(.96, 2))))

cat("3. Non-medical, illustrative numbers: spam filter")
cat("10,000 emails; filter flagged 480, of which 450 were spam; vendor reports recall = 90%\n")
print(complete_2x2(TP = 450, FP = 30, N = 10000, reported = list(Se = c(.90, 2))))

cat("4. Svirsky: only the positive row, N unknown (upper bound 1500)")
print(complete_2x2(TP = 93, FP = 150, N_max = 1500))

cat("5. Prior learned from complete tables in the same field (synthetic check)")
set.seed(1); ref <- t(replicate(30, rmultinom(1, 400, c(.20, .05, .02, .73))[, 1]))
colnames(ref) <- c("TP","FP","FN","TN"); a <- fit_prior_from_tables(ref); print(round(a, 1))
print(complete_2x2(TP = 81, FP = 19, N = 400, prior = a), top = 3)
