"""
===============================================================================
Maternal PUFA Intake and Infant Brain Development — HMinds Cohort
Public Analysis Script (Main Hypothesis Testing)

Posner et al. — Associations Between Maternal Polyunsaturated Fatty Acid Intake
and Infant Brain Structure

This script replicates the primary structural brain volume analyses reported in
the manuscript. It uses Multiple Imputation by Chained Equations (MICE) with
10 imputed datasets and pools results using Rubin's rules with Barnard-Rubin
degrees of freedom adjustment.

Requirements:
  pip install pandas numpy scipy statsmodels miceforest

Usage:
  1. Update DATA_PATH and RESULTS_DIR below to match your local file paths
  2. Run: python HMinds_PUFA_Analysis_public.py
===============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t as t_dist, norm as norm_dist
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import multipletests
import miceforest as mf
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# USER CONFIGURATION — UPDATE THESE PATHS
# ============================================================================

# Path to the master CSV file containing the HMinds PUFA dataset
DATA_PATH = '<PATH_TO_DATA>/HMinds_PUFA_MASTER.csv'

# Directory where results CSVs will be saved
RESULTS_DIR = '<PATH_TO_RESULTS>/'

# ============================================================================
# VERIFY PATHS
# ============================================================================
assert os.path.isfile(DATA_PATH), f"Data file not found: {DATA_PATH}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# Reproducibility: set random seed for MICE imputation and bootstrap
RANDOM_SEED = 123
MICE_M = 10        # Number of imputed datasets
MICE_MAXIT = 5     # Number of MICE iterations

np.random.seed(RANDOM_SEED)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset dimensions: {df.shape[0]} participants × {df.shape[1]} variables")

# --- Recode BirthType to binary ---
# Original values: "Vaginal", "1" (=Vaginal), "C-section", "2" (=C-section), "Forceps"
# Binary: 0 = vaginal delivery, 1 = non-vaginal delivery (C-section or forceps)
bt_map = {"Vaginal": 0, "1": 0, "C-section": 1, "2": 1, "Forceps": 1}
df["DeliveryNonVaginal"] = df["BirthType"].map(bt_map)

# --- Recode InfantSex to numeric ---
# F = 0, M = 1
df["InfantSex_num"] = df["InfantSex"].map({"F": 0, "M": 1}).astype(float)

# ============================================================================
# VARIABLE DEFINITIONS
# ============================================================================

# Primary predictor: maternal PUFA intake from 24-hour dietary recall
predictor = "nb_fa_poly"

# Covariates for structural brain volume models
# InfantSex_num: infant sex (0=female, 1=male)
# MomAgeEnroll: maternal age at enrollment (years)
# PrenatalBMI: maternal pre-pregnancy BMI (kg/m²)
# BirthWeight: infant birth weight (grams)
# PMA: postmenstrual age at MRI scan (weeks)
# TBV: total brain volume (mm³), controls for head size
# nb_energy_kcal: total energy intake (kcal), adjusts PUFA for diet size
# epds_score: Edinburgh Postnatal Depression Scale score
# SES: socioeconomic status (categorical, dummy-coded)
# DeliveryNonVaginal: delivery type (0=vaginal, 1=non-vaginal)
covariates = ["InfantSex_num", "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
              "PMA", "TBV", "nb_energy_kcal", "epds_score", "SES",
              "DeliveryNonVaginal"]

# Eight bilateral subcortical brain regions (volumes in mm³)
structural_outcomes = [
    "LeftHippocampus", "RightHippocampus",
    "LeftAmygdala", "RightAmygdala",
    "LeftCaudate", "RightCaudate",
    "LeftPutamen", "RightPutamen"
]

# Continuous variables that are z-scored prior to regression
# (standardized to mean=0, SD=1 so betas are standardized coefficients)
continuous_vars = [predictor, "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
                   "PMA", "TBV", "nb_energy_kcal", "epds_score"]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def z_score(x):
    """
    Standardize a variable to mean=0, SD=1.
    Uses ddof=1 (sample SD) to match R's scale() function.
    """
    return (x - x.mean()) / x.std(ddof=1)


def prepare_design_matrix(df_in, predictor, covariates, ses_col="SES"):
    """
    Build the OLS design matrix with:
      - Predictor (continuous, z-scored)
      - Continuous covariates (z-scored)
      - Binary covariates (InfantSex_num, DeliveryNonVaginal)
      - SES dummy variables (drop_first=True for reference category)
      - Intercept term
    """
    non_ses_covs = [c for c in covariates if c != ses_col]
    X = df_in[[predictor] + non_ses_covs].astype(float)
    ses_dummies = pd.get_dummies(df_in[ses_col], prefix="SES", drop_first=True).astype(float)
    X = pd.concat([X, ses_dummies], axis=1)
    X = add_constant(X)
    return X


def rubin_pool(betas, ses, n_avg, k):
    """
    Pool estimates from multiply imputed datasets using Rubin's rules.

    Parameters:
        betas: list of m coefficient estimates
        ses: list of m standard errors
        n_avg: average sample size across imputations
        k: number of parameters in the model

    Returns:
        Q_bar: pooled estimate
        se_pooled: pooled standard error
        t_stat: t-statistic
        p_val: two-sided p-value
        v_adj: Barnard-Rubin adjusted degrees of freedom
    """
    m = len(betas)

    # Pooled estimate (average across imputations)
    Q_bar = np.mean(betas)

    # Within-imputation variance (average of squared SEs)
    U_bar = np.mean(np.array(ses) ** 2)

    # Between-imputation variance
    B = np.var(betas, ddof=1)

    # Total variance (Rubin's combination rule)
    T = U_bar + (1 + 1/m) * B
    se_pooled = np.sqrt(T)

    # t-statistic
    t_stat = Q_bar / se_pooled if se_pooled > 0 else np.nan

    # Barnard-Rubin degrees of freedom adjustment
    r = (1 + 1/m) * B / U_bar if U_bar > 0 else 0
    v_old = (m - 1) * (1 + 1/r)**2 if r > 0 else np.inf
    v_obs_num = (n_avg - k + 1)
    v_obs = (v_obs_num / (v_obs_num + 2)) * v_obs_num * (1 - r / (r + 1)) if r > -1 else np.inf
    v_adj = (v_old * v_obs) / (v_old + v_obs) if (v_old + v_obs) > 0 else np.inf

    # Two-sided p-value
    if np.isfinite(v_adj) and v_adj > 0:
        p_val = 2 * t_dist.sf(abs(t_stat), v_adj)
    else:
        p_val = 2 * norm_dist.sf(abs(t_stat))

    return Q_bar, se_pooled, t_stat, p_val, v_adj


# ============================================================================
# MAIN ANALYSIS: STRUCTURAL BRAIN VOLUMES (MICE IMPUTED)
# ============================================================================
print("\n" + "=" * 70)
print("PRIMARY ANALYSIS: Maternal PUFA Intake → Subcortical Brain Volumes")
print("  Method: OLS regression with MICE imputation (m=10, maxit=5)")
print("  Correction: Benjamini-Hochberg FDR across 8 brain regions")
print("=" * 70)

# --- Step 1: Prepare analysis dataset ---
# Include only participants with the predictor and at least one structural outcome
all_analysis_cols = [predictor] + covariates + structural_outcomes
df_analysis = df[all_analysis_cols].copy()
df_analysis = df_analysis.dropna(subset=[predictor, structural_outcomes[0]]).reset_index(drop=True)

# SES must be categorical for miceforest imputation
df_analysis["SES"] = df_analysis["SES"].astype("category")

print(f"\nAnalysis sample: N = {len(df_analysis)}")
print(f"  (participants with dietary recall data and ≥1 structural MRI outcome)")

# --- Step 2: Multiple imputation ---
# miceforest uses random forest imputation (analogous to R mice PMM)
print("\nRunning MICE imputation...")
kernel = mf.ImputationKernel(df_analysis, num_datasets=MICE_M, random_state=RANDOM_SEED)
kernel.mice(MICE_MAXIT, verbose=False)
print(f"  Created {MICE_M} imputed datasets with {MICE_MAXIT} iterations each")

# --- Step 3: Fit models and pool results ---
print("\nFitting OLS models across imputed datasets...")
results = {}

for outcome in structural_outcomes:
    betas, ses_list, ns = [], [], []

    for i in range(MICE_M):
        imp_df = kernel.complete_data(i).copy()
        imp_df = imp_df.dropna(subset=[outcome])

        # Z-score all continuous variables (predictor, covariates, outcome)
        # This produces standardized betas (effect size in SD units)
        for v in continuous_vars + [outcome]:
            if v in imp_df.columns:
                imp_df[v] = z_score(imp_df[v])

        # Build design matrix and fit OLS
        X = prepare_design_matrix(imp_df, predictor, covariates)
        y = imp_df[outcome].astype(float)
        model = sm.OLS(y, X).fit()

        betas.append(model.params[predictor])
        ses_list.append(model.bse[predictor])
        ns.append(len(imp_df))

    # Pool across imputations using Rubin's rules
    n_avg = int(np.mean(ns))
    k = X.shape[1]  # number of model parameters
    Q, se, t, p, df_val = rubin_pool(betas, ses_list, n_avg, k)
    results[outcome] = {"N": n_avg, "beta": Q, "SE": se, "T": t, "P": p, "DF": df_val}

# --- Step 4: FDR correction ---
p_vals = [results[o]["P"] for o in structural_outcomes]
_, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
for i, o in enumerate(structural_outcomes):
    results[o]["P_FDR"] = p_fdr[i]

# --- Step 5: Display and save results ---
print(f"\n{'Region':<22} {'N':>4} {'β':>8} {'SE':>8} {'P':>8} {'P_FDR':>8}")
print("-" * 62)
for o in structural_outcomes:
    r = results[o]
    sig = " *" if r["P_FDR"] < 0.05 else ""
    print(f"{o:<22} {r['N']:>4} {r['beta']:>8.3f} {r['SE']:>8.3f} {r['P']:>8.3f} {r['P_FDR']:>8.3f}{sig}")

print("\n  * Survives FDR correction at q < 0.05")
print(f"\n  Note: β values are standardized coefficients (both predictor and")
print(f"  outcome z-scored). A β of -0.20 means a 1-SD increase in maternal")
print(f"  PUFA intake is associated with a 0.20-SD decrease in brain volume.")

# Save to CSV
results_df = pd.DataFrame([{"Outcome": o, **results[o]} for o in structural_outcomes])
outpath = os.path.join(RESULTS_DIR, "results_structural.csv")
results_df.to_csv(outpath, index=False)
print(f"\nResults saved to: {outpath}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
