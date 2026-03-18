# ============================================================================
# Maternal Prenatal Polyunsaturated Fatty Acid Intake and Infant Brain
# Development: A Prospective Cohort Study
#
# Posner J, Fay M, Michael A, Silva I, Abdala N, Coelho Milani AC,
# Oliveira Santana V, Shibuya Alves B, Wang Y, Duarte CS, Jackowski A.
# PLOS ONE, 2026.
#
# This script reproduces the primary hypothesis tests reported in the paper:
#   1. Structural MRI: PUFA intake -> subcortical volumes (8 regions, FDR)
#   2. Robust regression confirmation of significant findings
#   3. Functional connectivity: PUFA intake -> subcortical-prefrontal FC (32 tests, FDR)
#   4. Behavioral correlations: subcortical volumes -> CBCL scales
#   5. Mediation: PUFA -> right amygdala -> CBCL Stress Problems
#
# Missing covariates are handled via multiple imputation (MICE, m = 10).
# All continuous variables are z-scored prior to analysis.
# FDR correction is applied separately to structural (8 tests) and FC (32 tests).
#
# Requirements: R >= 4.0, packages: mice, MASS, dplyr, purrr, broom
# ============================================================================

# --- USER: Set paths --------------------------------------------------------
DATA_FILE  <- "hminds_PUFA_cbcl_MASTER.csv"   # path to your data file
OUTPUT_DIR <- "Results"                         # output folder for CSVs
# ----------------------------------------------------------------------------

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# install.packages(c("mice", "MASS", "dplyr", "purrr", "broom"))
library(mice)
library(MASS)
library(dplyr)
library(purrr)
library(broom)

select <- dplyr::select
filter <- dplyr::filter

set.seed(123)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

z_score <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

#' Pool regression results across multiply-imputed datasets using Rubin's rules
#' with Barnard-Rubin adjusted degrees of freedom.
#'
#' @param mids_obj  A mice::mids object (multiply-imputed datasets)
#' @param outcomes  Character vector of outcome variable names
#' @param predictor Character string naming the predictor of interest
#' @param covars    Character vector of covariate names
#' @return Data frame with columns: outcome, estimate, se, t_stat, p_value, p_FDR
pool_regression <- function(mids_obj, outcomes, predictor, covars) {
  m <- mids_obj$m
  results <- list()

  for (outcome in outcomes) {
    fm <- as.formula(paste(outcome, "~", predictor, "+",
                           paste(covars, collapse = " + ")))
    coefs <- ses <- numeric(m)

    for (k in seq_len(m)) {
      fit_k <- lm(fm, data = complete(mids_obj, k))
      s_k   <- summary(fit_k)$coefficients
      coefs[k] <- s_k[predictor, "Estimate"]
      ses[k]   <- s_k[predictor, "Std. Error"]
    }

    # Rubin's rules
    Q_bar <- mean(coefs)                          # pooled estimate
    U_bar <- mean(ses^2)                          # within-imputation variance
    B     <- var(coefs)                           # between-imputation variance
    T_var <- U_bar + B + B / m                    # total variance
    se_total <- sqrt(T_var)
    t_stat   <- Q_bar / se_total

    # Barnard-Rubin df adjustment
    lambda <- (B + B / m) / T_var
    df_old <- (m - 1) / lambda^2
    n_obs  <- nrow(complete(mids_obj, 1))
    p_covars <- length(covars) + 1                # +1 for predictor
    df_obs <- (n_obs - p_covars - 1) * (1 - lambda) *
      ((n_obs - p_covars - 1 + 1) / (n_obs - p_covars - 1 + 3))
    df_adj <- (df_old * df_obs) / (df_old + df_obs)

    p_val <- 2 * (1 - pt(abs(t_stat), df = df_adj))

    results[[outcome]] <- data.frame(
      outcome  = outcome,
      estimate = Q_bar,
      se       = se_total,
      t_stat   = t_stat,
      p_value  = p_val,
      stringsAsFactors = FALSE
    )
  }

  result_df <- do.call(rbind, results)
  rownames(result_df) <- NULL
  result_df$p_FDR <- p.adjust(result_df$p_value, method = "fdr")
  return(result_df)
}


# ============================================================================
# 1. DATA LOADING AND VARIABLE DEFINITIONS
# ============================================================================

cat("\n=== Loading data ===\n")
df <- read.csv(DATA_FILE, stringsAsFactors = FALSE)
cat("  Rows:", nrow(df), "| Columns:", ncol(df), "\n")

# Predictor: total polyunsaturated fatty acid intake (g/day), z-scored
predictor <- "nb_fa_poly"

# Structural outcomes: bilateral volumes of 4 subcortical regions (8 tests)
structural_outcomes <- c(
  "LeftHippocampus",  "RightHippocampus",
  "LeftAmygdala",     "RightAmygdala",
  "LeftCaudate",      "RightCaudate",
  "LeftPutamen",      "RightPutamen"
)

# Functional connectivity outcomes: 4 subcortical seeds x 4 prefrontal targets
# x 2 hemispheres (32 tests)
fc_outcomes <- c(
  "LHip_LLatOrb",  "LHip_LMedOrb",  "LHip_LParOrb",  "LHip_LRosAntCing",
  "LAmy_LLatOrb",  "LAmy_LMedOrb",  "LAmy_LParOrb",  "LAmy_LRosAntCing",
  "RHip_RLatOrb",  "RHip_RMedOrb",  "RHip_RParOrb",  "RHip_RRosAntCing",
  "RAmy_RLatOrb",  "RAmy_RMedOrb",  "RAmy_RParOrb",  "RAmy_RRosAntCing",
  "LCaud_L.lat.orb", "LCaud_L.med.orb", "LCaud_L.par.orb", "LCaud_L.ros.ant.cing",
  "LPut_L.lat.orb",  "LPut_L.med.orb",  "LPut_L.par.orb",  "LPut_L.ros.ant.cing",
  "RCaud_R.lat.orb", "RCaud_R.med.orb", "RCaud_R.par.orb", "RCaud_R.ros.ant.cing",
  "RPut_R.lat.orb",  "RPut_R.med.orb",  "RPut_R.par.orb",  "RPut_R.ros.ant.cing"
)

# Covariates for structural models
covariates_struct <- c(
  "InfantSex",            # categorical: M/F
  "MomAgeEnroll",         # maternal age at enrollment (years)
  "PrenatalBMI",          # pre-pregnancy BMI
  "BirthWeight",          # birth weight (grams)
  "GestAge_Recruitment",  # gestational age at recruitment (weeks)
  "PMA",                  # postmenstrual age at MRI scan (weeks)
  "TBV",                  # total brain volume (intracranial volume control)
  "nb_energy_kcal",       # total energy intake (kcal/day)
  "epds_score",           # Edinburgh Postnatal Depression Scale
  "SES"                   # socioeconomic status (categorical: A/B1/B2/C1/C2/DE)
)

# Covariates for FC models (TBV replaced with meanFD for motion control)
covariates_fc <- c(
  "InfantSex", "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
  "GestAge_Recruitment", "PMA", "nb_energy_kcal", "epds_score",
  "meanFD",               # mean framewise displacement (head motion)
  "SES"
)

# Continuous variables to z-score (categorical vars excluded)
continuous_struct <- c(predictor, "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
                       "GestAge_Recruitment", "PMA", "TBV", "nb_energy_kcal",
                       "epds_score")
continuous_fc    <- c(predictor, "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
                      "GestAge_Recruitment", "PMA", "nb_energy_kcal",
                      "epds_score", "meanFD")


# ============================================================================
# 2. PREPARE STRUCTURAL SAMPLE
# ============================================================================

cat("\n=== Preparing structural sample ===\n")

df_struct <- df %>%
  select(any_of(c(predictor, covariates_struct, structural_outcomes))) %>%
  filter(complete.cases(select(., all_of(c(predictor, structural_outcomes)))))
cat("  n =", nrow(df_struct),
    "(complete on predictor + all 8 structural volumes)\n")

# Z-score continuous variables and outcomes
for (v in continuous_struct) {
  if (v %in% names(df_struct)) df_struct[[v]] <- z_score(df_struct[[v]])
}
for (v in structural_outcomes) {
  df_struct[[v]] <- z_score(df_struct[[v]])
}


# ============================================================================
# 3. PREPARE FUNCTIONAL CONNECTIVITY SAMPLE
# ============================================================================

cat("\n=== Preparing FC sample ===\n")

df_fc <- df %>%
  select(any_of(c(predictor, covariates_fc, fc_outcomes))) %>%
  filter(complete.cases(select(., all_of(c(predictor, fc_outcomes)))))
cat("  n =", nrow(df_fc), "(complete on predictor + all 32 FC measures)\n")

for (v in continuous_fc) {
  if (v %in% names(df_fc)) df_fc[[v]] <- z_score(df_fc[[v]])
}
for (v in fc_outcomes) {
  if (v %in% names(df_fc)) df_fc[[v]] <- z_score(df_fc[[v]])
}


# ============================================================================
# 4. MULTIPLE IMPUTATION (MICE)
# ============================================================================

cat("\n=== Running MICE imputation (m = 10, maxit = 5) ===\n")

mids_struct <- mice(df_struct, m = 10, maxit = 5, seed = 123, printFlag = FALSE)
cat("  Structural imputation complete.\n")

mids_fc <- mice(df_fc, m = 10, maxit = 5, seed = 123, printFlag = FALSE)
cat("  FC imputation complete.\n")


# ============================================================================
# 5. PRIMARY ANALYSIS: STRUCTURAL MRI (Table 2 in paper)
#    8 OLS regressions, FDR-corrected across 8 tests
# ============================================================================

cat("\n=== Primary structural analysis (8 regions, FDR-corrected) ===\n")

struct_results <- pool_regression(mids_struct, structural_outcomes,
                                  predictor, covariates_struct)

cat(sprintf("\n  %-22s  %8s  %8s  %8s  %8s\n",
            "Region", "Beta", "SE", "P", "P_FDR"))
cat("  ", strrep("-", 60), "\n")
for (i in seq_len(nrow(struct_results))) {
  r <- struct_results[i, ]
  cat(sprintf("  %-22s  %8.4f  %8.4f  %8.4f  %8.4f  %s\n",
              r$outcome, r$estimate, r$se, r$p_value, r$p_FDR,
              ifelse(r$p_FDR < 0.05, "*", "")))
}

write.csv(struct_results,
          file.path(OUTPUT_DIR, "primary_structural.csv"), row.names = FALSE)


# ============================================================================
# 6. ROBUST REGRESSION CONFIRMATION (Huber M-estimator)
#    Applied to regions surviving FDR correction
# ============================================================================

sig_regions <- struct_results %>% filter(p_FDR < 0.05) %>% pull(outcome)

cat("\n=== Robust regression (FDR-significant regions) ===\n")

if (length(sig_regions) > 0) {
  m_imp <- mids_struct$m
  robust_results <- list()

  for (outcome in sig_regions) {
    fm <- as.formula(paste(outcome, "~", predictor, "+",
                           paste(covariates_struct, collapse = " + ")))
    coefs <- vars <- numeric(m_imp)

    for (k in seq_len(m_imp)) {
      fit_k <- rlm(fm, data = complete(mids_struct, k), method = "M")
      s_k   <- summary(fit_k)
      coefs[k] <- coef(fit_k)[predictor]
      vars[k]  <- s_k$coefficients[predictor, "Std. Error"]^2
    }

    Q_bar  <- mean(coefs)
    U_bar  <- mean(vars)
    B      <- var(coefs)
    Total  <- U_bar + B + B / m_imp
    t_stat <- Q_bar / sqrt(Total)
    df_den <- (m_imp - 1) * (1 + U_bar / (B + B / m_imp))^2
    p_val  <- 2 * (1 - pt(abs(t_stat), df_den))

    robust_results[[outcome]] <- data.frame(
      outcome = outcome, estimate = Q_bar, se = sqrt(Total),
      t_stat = t_stat, p_value = p_val, stringsAsFactors = FALSE
    )
    cat(sprintf("  %s: Beta = %.4f, SE = %.4f, P = %.4f\n",
                outcome, Q_bar, sqrt(Total), p_val))
  }

  robust_df <- do.call(rbind, robust_results)
  rownames(robust_df) <- NULL
  write.csv(robust_df,
            file.path(OUTPUT_DIR, "robust_regression.csv"), row.names = FALSE)
} else {
  cat("  No regions survived FDR correction.\n")
}


# ============================================================================
# 7. FUNCTIONAL CONNECTIVITY ANALYSIS (Table 3 in paper)
#    32 OLS regressions, FDR-corrected across 32 tests
# ============================================================================

cat("\n=== Functional connectivity analysis (32 connections, FDR-corrected) ===\n")

fc_results <- pool_regression(mids_fc, fc_outcomes, predictor, covariates_fc)

# Show top 10 by p-value
fc_sorted <- fc_results[order(fc_results$p_value), ]
cat(sprintf("\n  %-28s  %8s  %8s  %8s  %8s\n",
            "Connection", "Beta", "SE", "P", "P_FDR"))
cat("  ", strrep("-", 68), "\n")
for (i in seq_len(min(10, nrow(fc_sorted)))) {
  r <- fc_sorted[i, ]
  cat(sprintf("  %-28s  %8.4f  %8.4f  %8.4f  %8.4f\n",
              r$outcome, r$estimate, r$se, r$p_value, r$p_FDR))
}

write.csv(fc_results,
          file.path(OUTPUT_DIR, "functional_connectivity.csv"), row.names = FALSE)


# ============================================================================
# 8. BEHAVIORAL CORRELATIONS: RIGHT AMYGDALA x CBCL SCALES
# ============================================================================

cat("\n=== Behavioral correlations: Right Amygdala x CBCL scales ===\n")

cbcl_vars <- grep("^cbcl_.*_t$", names(df), value = TRUE, ignore.case = TRUE)
cat("  CBCL T-score variables:", length(cbcl_vars), "\n")

cbcl_amy_results <- list()
for (cbcl_var in cbcl_vars) {
  test <- cor.test(df[["RightAmygdala"]], df[[cbcl_var]], use = "complete.obs")
  cbcl_amy_results[[cbcl_var]] <- data.frame(
    cbcl_scale = cbcl_var,
    r = test$estimate,
    p_value = test$p.value,
    n = sum(!is.na(df[["RightAmygdala"]]) & !is.na(df[[cbcl_var]])),
    stringsAsFactors = FALSE
  )
}

cbcl_df <- do.call(rbind, cbcl_amy_results)
rownames(cbcl_df) <- NULL
cbcl_df <- cbcl_df[order(cbcl_df$p_value), ]

cat(sprintf("\n  %-30s  %8s  %8s  %4s\n", "CBCL Scale", "r", "P", "n"))
cat("  ", strrep("-", 55), "\n")
for (i in seq_len(nrow(cbcl_df))) {
  r <- cbcl_df[i, ]
  cat(sprintf("  %-30s  %8.4f  %8.4f  %4d  %s\n",
              r$cbcl_scale, r$r, r$p_value, r$n,
              ifelse(r$p_value < 0.05, "*", "")))
}

write.csv(cbcl_df,
          file.path(OUTPUT_DIR, "behavioral_correlations.csv"), row.names = FALSE)


# ============================================================================
# 9. MEDIATION: PUFA -> RIGHT AMYGDALA -> CBCL STRESS PROBLEMS
#    Sobel test + bootstrap 95% CI (5000 iterations)
# ============================================================================

cat("\n=== Mediation: PUFA -> Right Amygdala -> CBCL Stress Problems ===\n")

med_data <- df %>%
  select(nb_fa_poly, RightAmygdala, cbcl_stress_prob_t) %>%
  filter(complete.cases(.))

cat("  n =", nrow(med_data), "\n")

med_data$X  <- z_score(med_data$nb_fa_poly)
med_data$M  <- z_score(med_data$RightAmygdala)
med_data$Y  <- z_score(med_data$cbcl_stress_prob_t)

# Path a: X -> M (PUFA -> Amygdala)
fit_a <- lm(M ~ X, data = med_data)
a     <- coef(fit_a)["X"]
se_a  <- summary(fit_a)$coefficients["X", "Std. Error"]

# Path b: M -> Y | X (Amygdala -> CBCL, controlling for PUFA)
fit_b <- lm(Y ~ M + X, data = med_data)
b     <- coef(fit_b)["M"]
se_b  <- summary(fit_b)$coefficients["M", "Std. Error"]

# Direct effect (c')
c_prime <- coef(fit_b)["X"]

# Indirect effect and Sobel test
indirect    <- a * b
se_indirect <- sqrt(a^2 * se_b^2 + b^2 * se_a^2)
z_sobel     <- indirect / se_indirect
p_sobel     <- 2 * (1 - pnorm(abs(z_sobel)))

# Bootstrap confidence interval
set.seed(1234)
n_boot <- 5000
boot_indirect <- numeric(n_boot)
for (i in seq_len(n_boot)) {
  idx <- sample(nrow(med_data), replace = TRUE)
  d   <- med_data[idx, ]
  a_b <- coef(lm(M ~ X, data = d))["X"]
  b_b <- coef(lm(Y ~ M + X, data = d))["M"]
  boot_indirect[i] <- a_b * b_b
}
boot_ci <- quantile(boot_indirect, c(0.025, 0.975))

cat(sprintf("  Path a (PUFA -> Amygdala):   %.4f (SE = %.4f)\n", a, se_a))
cat(sprintf("  Path b (Amygdala -> Stress): %.4f (SE = %.4f)\n", b, se_b))
cat(sprintf("  Direct effect (c'):          %.4f\n", c_prime))
cat(sprintf("  Indirect effect (a x b):     %.4f\n", indirect))
cat(sprintf("  Sobel z = %.4f, P = %.4f\n", z_sobel, p_sobel))
cat(sprintf("  Bootstrap 95%% CI: [%.4f, %.4f]\n", boot_ci[1], boot_ci[2]))

med_summary <- data.frame(
  n = nrow(med_data), path_a = a, se_a = se_a,
  path_b = b, se_b = se_b, direct_effect = c_prime,
  indirect_effect = indirect, sobel_z = z_sobel, sobel_p = p_sobel,
  boot_ci_lower = boot_ci[1], boot_ci_upper = boot_ci[2],
  stringsAsFactors = FALSE
)
write.csv(med_summary,
          file.path(OUTPUT_DIR, "mediation.csv"), row.names = FALSE)


# ============================================================================
# SESSION INFO
# ============================================================================

cat("\n=== Session info ===\n")
sessionInfo()

cat("\n=== Analysis complete:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "===\n")
cat("Results saved to:", OUTPUT_DIR, "\n")
