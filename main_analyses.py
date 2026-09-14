"""
===============================================================================
Maternal PUFA Intake and Infant Brain Development - HMinds Cohort
Primary analysis script

Posner J, Fay M, Michael A, Silva I, Abdala N, Coelho Milani AC, Oliveira
Santana V, Shibuya Alves B, Wang Y, Duarte CS, Jackowski A.
"Maternal prenatal polyunsaturated fatty acid intake and infant brain
development: A prospective cohort study."

-------------------------------------------------------------------------------
SCOPE

This script reproduces the PRIMARY analyses reported in the manuscript:

  Section 1  Cohort descriptives                            -> Table 1
  Section 2  Structural volumes, MICE-imputed OLS + FDR     -> Table 2
  Section 3  Robust regression confirmation                 -> Results text
  Section 4  PUFA x infant sex interaction                  -> Results text
  Section 5  Functional connectivity, MICE + FDR            -> Results text
  Section 6  Exploratory CBCL correlations + bootstrap      -> Results text
  Section 7  Exploratory mediation                          -> Results text

Sensitivity and supplementary analyses (complete-case models, delivery-mode
subsamples, outlier-exclusion models, stricter motion thresholds) are not
included here; they reuse the same functions with a filtered input frame.

-------------------------------------------------------------------------------
DATA

No participant data are distributed with this code. The script expects a
single de-identified analysis file in CSV format, one row per mother-infant
dyad, with the column names listed in DATA_DICTIONARY.md. Supply its location
with --data; nothing about the file's contents is hard-coded.

De-identified data are available in anonymized form through the NIMH Data
Archive (nda.nih.gov) under collection C3811. Requests for additional
de-identified data may be directed to the corresponding author.

-------------------------------------------------------------------------------
REQUIREMENTS

  pip install -r requirements.txt

  Verified reproducing stack:
    python 3.10.12, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3,
    statsmodels 0.15.0, miceforest 6.0.5, lightgbm 4.5.0

  NOTE: miceforest 6.0.5 is incompatible with lightgbm >= 4.6, which raises
  "Booster.__inner_predict() takes 1 positional argument but 2 were given".
  Pin lightgbm==4.5.0.

-------------------------------------------------------------------------------
USAGE

  python main_analyses.py --data <path/to/analysis_dataset.csv> \
                          --results-dir <path/to/output/>

  Defaults are ./data/analysis_dataset.csv and ./results/.
===============================================================================
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant

import miceforest as mf

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_DATA = os.path.join("data", "analysis_dataset.csv")
DEFAULT_RESULTS = "results"

RANDOM_SEED = 123        # MICE imputation
BOOTSTRAP_SEED = 1234    # bootstrap resampling
MICE_M = 10              # number of imputed datasets
MICE_MAXIT = 5           # MICE iterations
N_BOOT_CBCL = 1000       # bootstrap replicates, CBCL correlation
N_BOOT_MEDIATION = 5000  # bootstrap replicates, mediation indirect effect

PREDICTOR = "nb_fa_poly"   # total maternal PUFA intake, g/day

STRUCTURAL_OUTCOMES = [
    "LeftHippocampus", "RightHippocampus",
    "LeftAmygdala", "RightAmygdala",
    "LeftCaudate", "RightCaudate",
    "LeftPutamen", "RightPutamen",
]

# Covariates for structural models.
#   InfantSex_num      infant sex (0=female, 1=male)
#   MomAgeEnroll       maternal age at enrollment (years)
#   PrenatalBMI        maternal pregestational BMI (kg/m2)
#   BirthWeight        infant birth weight (grams)
#   PMA                postmenstrual age at MRI scan (weeks)
#   TBV                total brain volume (mm3); controls for head size
#   nb_energy_kcal     total energy intake (kcal); adjusts PUFA for diet size
#   epds_score         Edinburgh Postnatal Depression Scale score
#   SES_bin            socioeconomic status (0=high/middle, 1=low)
#   DeliveryNonVaginal delivery type (0=vaginal, 1=non-vaginal)
COVARIATES_STRUCTURAL = [
    "InfantSex_num", "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
    "PMA", "TBV", "nb_energy_kcal", "epds_score", "SES_bin",
    "DeliveryNonVaginal",
]

# Connectivity models drop TBV (not a head-size question) and add mean
# framewise displacement to control for in-scanner motion.
COVARIATES_FC = [
    "InfantSex_num", "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
    "PMA", "nb_energy_kcal", "epds_score", "meanFD", "SES_bin",
    "DeliveryNonVaginal",
]

# 32 within-hemisphere subcortical-seed x prefrontal-target Fisher-z
# correlations. The camelCase / dotted naming split is inherited from the
# connectivity toolbox output and is preserved so the column names match.
FC_OUTCOMES = [
    "LHip_LLatOrb", "LHip_LMedOrb", "LHip_LParOrb", "LHip_LRosAntCing",
    "LAmy_LLatOrb", "LAmy_LMedOrb", "LAmy_LParOrb", "LAmy_LRosAntCing",
    "RHip_RLatOrb", "RHip_RMedOrb", "RHip_RParOrb", "RHip_RRosAntCing",
    "RAmy_RLatOrb", "RAmy_RMedOrb", "RAmy_RParOrb", "RAmy_RRosAntCing",
    "LCaud_L.lat.orb", "LCaud_L.med.orb", "LCaud_L.par.orb", "LCaud_L.ros.ant.cing",
    "LPut_L.lat.orb", "LPut_L.med.orb", "LPut_L.par.orb", "LPut_L.ros.ant.cing",
    "RCaud_R.lat.orb", "RCaud_R.med.orb", "RCaud_R.par.orb", "RCaud_R.ros.ant.cing",
    "RPut_R.lat.orb", "RPut_R.med.orb", "RPut_R.par.orb", "RPut_R.ros.ant.cing",
]

# Continuous variables z-scored before regression, so betas are standardized.
CONT_VARS_STRUCTURAL = [
    PREDICTOR, "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
    "PMA", "TBV", "nb_energy_kcal", "epds_score",
]
CONT_VARS_FC = [
    PREDICTOR, "MomAgeEnroll", "PrenatalBMI", "BirthWeight",
    "PMA", "nb_energy_kcal", "epds_score", "meanFD",
]

# Descriptive variables reported in Table 1.
TABLE1_CONTINUOUS = [
    ("GestAgeRecall", "Gestational age at dietary recall, wk"),
    ("PMA", "Postmenstrual age at scan, wk"),
    ("MomAgeEnroll", "Maternal age at enrollment, y"),
    ("BirthWeight", "Birth weight, g"),
    ("PrenatalBMI", "Pregestational BMI"),
    (PREDICTOR, "Maternal PUFA intake, g/d"),
    ("nb_energy_kcal", "Total energy intake, kcal/d"),
    ("epds_score", "EPDS depression score"),
]
TABLE1_CATEGORICAL = [
    ("InfantSex", "Infant sex at birth"),
    ("BirthType", "Mode of delivery"),
    ("SES", "Socioeconomic status (CCEB class)"),
]


# =============================================================================
# UTILITIES
# =============================================================================
def z_score(x):
    """Standardize to mean 0, SD 1. ddof=1 (sample SD) matches R's scale()."""
    return (x - x.mean()) / x.std(ddof=1)


def prepare_X(df_in, predictor, covariates):
    """Design matrix with intercept. SES_bin is already 0/1; no dummies needed."""
    X = df_in[[predictor] + covariates].astype(float)
    return add_constant(X)


def rubin_pool(betas, ses, n_avg, k):
    """
    Pool multiply-imputed estimates using Rubin's rules with the
    Barnard-Rubin degrees-of-freedom adjustment.

    Returns (pooled_estimate, pooled_se, t_statistic, p_value, adjusted_df).
    """
    m = len(betas)
    Q_bar = np.mean(betas)                      # pooled estimate
    U_bar = np.mean(np.array(ses) ** 2)         # within-imputation variance
    B = np.var(betas, ddof=1)                   # between-imputation variance
    T = U_bar + (1 + 1 / m) * B                 # total variance
    se_pooled = np.sqrt(T)
    t_stat = Q_bar / se_pooled if se_pooled > 0 else np.nan

    r = (1 + 1 / m) * B / U_bar if U_bar > 0 else 0
    v_old = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
    v_obs = (((n_avg - k + 1) / (n_avg - k + 3)) * (n_avg - k + 1)
             * (1 - r / (r + 1))) if r > -1 else np.inf
    v_adj = (v_old * v_obs) / (v_old + v_obs) if (v_old + v_obs) > 0 else np.inf

    if np.isfinite(v_adj) and v_adj > 0:
        p_val = 2 * t_dist.sf(abs(t_stat), v_adj)
    else:
        p_val = 2 * norm_dist.sf(abs(t_stat))
    return Q_bar, se_pooled, t_stat, p_val, v_adj


def run_mice_analysis(df_input, predictor, covariates, outcomes, cont_z_vars,
                      m=MICE_M, maxit=MICE_MAXIT, seed=RANDOM_SEED):
    """
    MICE-imputed OLS for a family of outcomes, pooled by Rubin's rules.

    The analysis sample is defined by dropping rows missing the predictor or
    the FIRST outcome in `outcomes`. In this cohort that is equivalent to
    requiring all outcomes in the family: every dyad with any usable
    structural MRI has all 8 volumes, and every dyad with usable rs-fMRI has
    all 32 connectivity values. A replicator applying this code to data where
    that equivalence does not hold would obtain a different per-outcome sample
    and should switch to an outcome-wise dropna before imputation.

    Imputation runs on the restricted sample, so any subsetting done by the
    caller correctly happens BEFORE imputation rather than after it.
    """
    all_cols = [predictor] + covariates + outcomes
    df_work = (df_input[all_cols]
               .dropna(subset=[predictor, outcomes[0]])
               .reset_index(drop=True))
    print(f"  Sample entering imputation: N = {len(df_work)}")

    kernel = mf.ImputationKernel(df_work, num_datasets=m, random_state=seed)
    kernel.mice(maxit, verbose=False)

    results = {}
    for outcome in outcomes:
        betas, ses_list, ns = [], [], []
        for i in range(m):
            imp_df = kernel.complete_data(i).copy().dropna(subset=[outcome])
            # z-score within each imputed dataset, after outcome-wise dropna,
            # so betas are standardized on the per-outcome analytic sample
            for v in cont_z_vars + [outcome]:
                if v in imp_df.columns:
                    imp_df[v] = z_score(imp_df[v])
            X = prepare_X(imp_df, predictor, covariates)
            model = sm.OLS(imp_df[outcome].astype(float), X).fit()
            betas.append(model.params[predictor])
            ses_list.append(model.bse[predictor])
            ns.append(len(imp_df))
        n_avg = int(np.mean(ns))
        k = X.shape[1]
        Q, se, t, p, df_val = rubin_pool(betas, ses_list, n_avg, k)
        results[outcome] = {"N": n_avg, "beta": Q, "SE": se,
                            "T": t, "P": p, "DF": df_val}
    return results


def add_ci(frame, beta_col="Beta", se_col="SE"):
    """95% CI as beta +/- 1.96*SE, matching the published tables."""
    frame["CI_Lower"] = frame[beta_col] - 1.96 * frame[se_col]
    frame["CI_Upper"] = frame[beta_col] + 1.96 * frame[se_col]
    return frame


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_and_recode(data_path):
    """Load the analysis file and derive the three recoded model variables."""
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} dyads x {df.shape[1]} variables")

    # Delivery type -> binary. The map accepts both the string labels and the
    # legacy numeric codes. Unmapped or blank values become NaN and are imputed.
    bt_map = {"Vaginal": 0, "1": 0, "C-section": 1, "2": 1, "Forceps": 1}
    df["DeliveryNonVaginal"] = df["BirthType"].map(bt_map)

    # Infant sex -> numeric
    df["InfantSex_num"] = df["InfantSex"].map({"F": 0, "M": 1}).astype(float)

    # SES -> binary. SES_binary is derived from the Brazilian Economic
    # Classification Criteria (CCEB) class: A1/B1/B2/C1 = "High", C2/DE = "Low".
    # The full six-level parameterization produced elevated variance inflation
    # factors and was collapsed, as reported in the manuscript.
    # NOTE: this equality test maps a MISSING SES_binary to 0 (high/middle)
    # rather than NaN, so SES is never imputed. Flagged for anyone reusing
    # this code on a different extract.
    df["SES_bin"] = (df["SES_binary"] == "Low").astype(float)
    return df


# =============================================================================
# MAIN
# =============================================================================
def main(data_path, results_dir):
    assert os.path.isfile(data_path), (
        f"Data file not found: {data_path}\n"
        "Supply the path to the de-identified analysis CSV with --data. "
        "See DATA_DICTIONARY.md for the expected columns.")
    os.makedirs(results_dir, exist_ok=True)
    out = lambda name: os.path.join(results_dir, name)

    np.random.seed(RANDOM_SEED)
    df = load_and_recode(data_path)

    # -------------------------------------------------------------------------
    # SECTION 1: COHORT DESCRIPTIVES  ->  Table 1
    # -------------------------------------------------------------------------
    banner("SECTION 1: COHORT DESCRIPTIVES  ->  Table 1")
    # Descriptives are computed on the primary structural analytic sample.
    analytic = df.dropna(subset=[PREDICTOR, STRUCTURAL_OUTCOMES[0]])
    print(f"Primary analytic sample: N = {len(analytic)}")

    desc_rows = []
    for col, label in TABLE1_CONTINUOUS:
        if col not in analytic.columns:
            continue
        s = analytic[col].dropna()
        desc_rows.append({"Variable": label, "Type": "continuous", "N": len(s),
                          "Mean": s.mean(), "SD": s.std(ddof=1),
                          "Min": s.min(), "Max": s.max()})
    for col, label in TABLE1_CATEGORICAL:
        if col not in analytic.columns:
            continue
        counts = analytic[col].value_counts(dropna=True)
        for level, n in counts.items():
            desc_rows.append({"Variable": f"{label}: {level}",
                              "Type": "categorical", "N": int(n),
                              "Mean": np.nan, "SD": np.nan,
                              "Min": np.nan, "Max": np.nan})
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(out("results_descriptives.csv"), index=False)
    print(desc_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # SECTION 2: STRUCTURAL VOLUMES (MICE)  ->  Table 2
    # -------------------------------------------------------------------------
    banner("SECTION 2: STRUCTURAL ANALYSIS (MICE IMPUTED)  ->  Table 2")
    structural = run_mice_analysis(df, PREDICTOR, COVARIATES_STRUCTURAL,
                                   STRUCTURAL_OUTCOMES, CONT_VARS_STRUCTURAL)
    # Benjamini-Hochberg FDR across the 8 structural tests only. The
    # connectivity tests form a separate family, as stated in the Methods.
    _, p_fdr, _, _ = multipletests(
        [structural[o]["P"] for o in STRUCTURAL_OUTCOMES], method="fdr_bh")
    for i, o in enumerate(STRUCTURAL_OUTCOMES):
        structural[o]["P_FDR"] = p_fdr[i]

    struct_df = pd.DataFrame([{"Outcome": o, **structural[o]}
                              for o in STRUCTURAL_OUTCOMES])
    struct_df = add_ci(struct_df, "beta", "SE")
    struct_df.to_csv(out("results_structural.csv"), index=False)

    print(f"\n{'Region':<20} {'N':>4} {'beta':>8} {'SE':>8} {'P':>8} {'P_FDR':>8}")
    print("-" * 60)
    for o in STRUCTURAL_OUTCOMES:
        r = structural[o]
        print(f"{o:<20} {r['N']:>4} {r['beta']:>8.3f} {r['SE']:>8.3f} "
              f"{r['P']:>8.3f} {r['P_FDR']:>8.3f} {'*' if r['P_FDR'] < 0.05 else ''}")
    print("\n  * survives FDR correction at q < 0.05")
    print("  beta is standardized: a 1-SD increase in maternal PUFA intake is")
    print("  associated with a beta-SD change in regional volume.")

    # -------------------------------------------------------------------------
    # SECTION 3: ROBUST REGRESSION CONFIRMATION  ->  Results text
    # -------------------------------------------------------------------------
    banner("SECTION 3: ROBUST REGRESSION (Huber M-estimator)")
    # Applied to whichever outcomes survive FDR above, selected programmatically.
    # This is a COMPLETE-CASE model, so its sample is smaller than the
    # MICE-imputed primary model and the coefficients are not directly
    # comparable in magnitude.
    fdr_sig = [o for o in STRUCTURAL_OUTCOMES if structural[o]["P_FDR"] < 0.05]
    print(f"FDR-significant outcomes: {fdr_sig}")

    robust_rows = []
    if fdr_sig:
        df_robust = df[[PREDICTOR] + COVARIATES_STRUCTURAL + fdr_sig].dropna().copy()
        print(f"Complete cases: {len(df_robust)}")
        for v in CONT_VARS_STRUCTURAL + fdr_sig:
            if v in df_robust.columns:
                df_robust[v] = z_score(df_robust[v])
        for outcome in fdr_sig:
            X = prepare_X(df_robust, PREDICTOR, COVARIATES_STRUCTURAL)
            model = RLM(df_robust[outcome].astype(float), X, M=HuberT()).fit()
            robust_rows.append({"Outcome": outcome, "N": len(df_robust),
                                "Beta": model.params[PREDICTOR],
                                "SE": model.bse[PREDICTOR],
                                "T": model.tvalues[PREDICTOR],
                                "P": model.pvalues[PREDICTOR]})
            print(f"  {outcome}: beta={model.params[PREDICTOR]:.3f}, "
                  f"P={model.pvalues[PREDICTOR]:.3f}")
        add_ci(pd.DataFrame(robust_rows)).to_csv(out("results_robust.csv"),
                                                 index=False)

    # -------------------------------------------------------------------------
    # SECTION 4: PUFA x INFANT SEX INTERACTION  ->  Results text
    # -------------------------------------------------------------------------
    banner("SECTION 4: INTERACTION ANALYSIS (PUFA x InfantSex)")
    df_int = (df[[PREDICTOR] + COVARIATES_STRUCTURAL + STRUCTURAL_OUTCOMES]
              .dropna(subset=[PREDICTOR, STRUCTURAL_OUTCOMES[0]])
              .reset_index(drop=True))
    print(f"  Sample entering imputation: N = {len(df_int)}")
    kernel_int = mf.ImputationKernel(df_int, num_datasets=MICE_M,
                                     random_state=RANDOM_SEED)
    kernel_int.mice(MICE_MAXIT, verbose=False)

    interaction_rows = []
    for outcome in STRUCTURAL_OUTCOMES:
        betas, ses_list, ns = [], [], []
        for i in range(MICE_M):
            imp_df = kernel_int.complete_data(i).copy().dropna(subset=[outcome])
            for v in CONT_VARS_STRUCTURAL + [outcome]:
                if v in imp_df.columns:
                    imp_df[v] = z_score(imp_df[v])
            # interaction built AFTER z-scoring: z(PUFA) x 0/1 sex, so the
            # product term is not itself re-standardized
            imp_df["PUFA_x_Sex"] = imp_df[PREDICTOR] * imp_df["InfantSex_num"]
            covs = COVARIATES_STRUCTURAL + ["PUFA_x_Sex"]
            X = add_constant(imp_df[[PREDICTOR] + covs].astype(float))
            model = sm.OLS(imp_df[outcome].astype(float), X).fit()
            betas.append(model.params["PUFA_x_Sex"])
            ses_list.append(model.bse["PUFA_x_Sex"])
            ns.append(len(imp_df))
        n_avg = int(np.mean(ns))
        Q, se, t, p, _ = rubin_pool(betas, ses_list, n_avg, X.shape[1])
        interaction_rows.append({"Outcome": outcome, "N": n_avg, "Beta": Q,
                                 "SE": se, "T": t, "P": p})
    interact_df = add_ci(pd.DataFrame(interaction_rows))
    interact_df.to_csv(out("results_interaction.csv"), index=False)
    print(interact_df[["Outcome", "Beta", "P"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # SECTION 5: FUNCTIONAL CONNECTIVITY (MICE)  ->  Results text
    # -------------------------------------------------------------------------
    banner("SECTION 5: FC ANALYSIS (MICE IMPUTED)")
    fc = run_mice_analysis(df, PREDICTOR, COVARIATES_FC, FC_OUTCOMES,
                           CONT_VARS_FC)
    # Separate BH FDR family: the 32 connectivity tests only.
    _, p_fdr_fc, _, _ = multipletests([fc[o]["P"] for o in FC_OUTCOMES],
                                      method="fdr_bh")
    for i, o in enumerate(FC_OUTCOMES):
        fc[o]["P_FDR"] = p_fdr_fc[i]
    fc_df = add_ci(pd.DataFrame([{"Outcome": o, **fc[o]} for o in FC_OUTCOMES]),
                   "beta", "SE")
    fc_df.to_csv(out("results_fc.csv"), index=False)
    print(f"  Minimum P_FDR across 32 connections: {min(p_fdr_fc):.3f}")

    # -------------------------------------------------------------------------
    # SECTION 6: EXPLORATORY CBCL CORRELATIONS  ->  Results text
    # -------------------------------------------------------------------------
    banner("SECTION 6: EXPLORATORY CBCL CORRELATIONS")
    # These are bivariate Pearson correlations: no covariates, no imputation,
    # and no multiplicity correction. They are reported as exploratory.
    #
    # Column-naming caution: the "_s" suffix holds CBCL T scores, which is the
    # standard reporting metric and what all analyses use. A "_t" suffix, if
    # present in the source extract, holds raw item totals.
    cbcl_cols = [c for c in df.columns
                 if c.startswith("cbcl_") and c.endswith("_s")]
    print(f"CBCL T-score scales: {len(cbcl_cols)}")

    def bivariate(x_col, cols):
        rows = []
        for c in cols:
            pair = df[[x_col, c]].dropna()
            if len(pair) > 2:
                r, p = stats.pearsonr(pair[x_col], pair[c])
                rows.append({"CBCL": c, "r": r, "P": p, "N": len(pair)})
        return pd.DataFrame(rows)

    bivariate(PREDICTOR, cbcl_cols).to_csv(out("results_cbcl_pufa.csv"),
                                           index=False)
    amyg_cbcl = bivariate("RightAmygdala", cbcl_cols)
    amyg_cbcl.to_csv(out("results_cbcl_amygdala.csv"), index=False)
    print(amyg_cbcl.sort_values("P").head(3).to_string(index=False))

    # Percentile bootstrap of the amygdala-CBCL correlations reported in text.
    # The seed is reset per scale so each interval is independently reproducible.
    banner("SECTION 6b: BOOTSTRAP OF AMYGDALA-CBCL CORRELATIONS")
    boot_rows = []
    for scale in ["cbcl_stress_prob_s", "cbcl_anxious_depres_s"]:
        if scale not in df.columns:
            continue
        pair = df[["RightAmygdala", scale]].dropna()
        np.random.seed(BOOTSTRAP_SEED)
        rs = []
        for _ in range(N_BOOT_CBCL):
            idx = np.random.choice(len(pair), size=len(pair), replace=True)
            samp = pair.iloc[idx]
            rs.append(stats.pearsonr(samp["RightAmygdala"], samp[scale])[0])
        rs = np.array(rs)
        lo, hi = np.percentile(rs, [2.5, 97.5])
        boot_rows.append({"Scale": scale, "Sample_N": len(pair),
                          "N_Bootstrap": N_BOOT_CBCL, "Mean_r": rs.mean(),
                          "CI_Lower": lo, "CI_Upper": hi})
        print(f"  {scale}: mean r={rs.mean():.4f}, "
              f"95% CI [{lo:.4f}, {hi:.4f}], n={len(pair)}")
    pd.DataFrame(boot_rows).to_csv(out("results_bootstrap.csv"), index=False)

    # -------------------------------------------------------------------------
    # SECTION 7: EXPLORATORY MEDIATION  ->  Results text
    # -------------------------------------------------------------------------
    banner("SECTION 7: MEDIATION (PUFA -> RightAmygdala -> CBCL Stress)")
    # Unadjusted three-variable path model; no covariates, unlike the
    # covariate-adjusted primary models above.
    df_med = df[[PREDICTOR, "RightAmygdala", "cbcl_stress_prob_s"]].dropna().copy()
    print(f"Mediation N: {len(df_med)}")
    for col in df_med.columns:
        df_med[col] = z_score(df_med[col])

    model_a = sm.OLS(df_med["RightAmygdala"],
                     add_constant(df_med[[PREDICTOR]])).fit()
    a, se_a = model_a.params[PREDICTOR], model_a.bse[PREDICTOR]
    model_b = sm.OLS(df_med["cbcl_stress_prob_s"],
                     add_constant(df_med[["RightAmygdala", PREDICTOR]])).fit()
    b, se_b = model_b.params["RightAmygdala"], model_b.bse["RightAmygdala"]
    c_prime = model_b.params[PREDICTOR]

    indirect = a * b
    sobel_se = np.sqrt((a ** 2) * (se_b ** 2) + (b ** 2) * (se_a ** 2))
    z_sobel = indirect / sobel_se
    p_sobel = 2 * norm_dist.sf(abs(z_sobel))
    print(f"Path a: {a:.4f}, Path b: {b:.4f}, c': {c_prime:.4f}")
    print(f"Indirect: {indirect:.4f}, Sobel P: {p_sobel:.4f}")

    # Percentile bootstrap of the indirect effect (not bias-corrected).
    np.random.seed(BOOTSTRAP_SEED)
    boots = []
    for _ in range(N_BOOT_MEDIATION):
        d = df_med.iloc[np.random.choice(len(df_med), size=len(df_med),
                                         replace=True)]
        m_a = sm.OLS(d["RightAmygdala"], add_constant(d[[PREDICTOR]])).fit()
        m_b = sm.OLS(d["cbcl_stress_prob_s"],
                     add_constant(d[["RightAmygdala", PREDICTOR]])).fit()
        boots.append(m_a.params[PREDICTOR] * m_b.params["RightAmygdala"])
    boots = np.array(boots)
    med_ci = np.percentile(boots, [2.5, 97.5])
    print(f"Bootstrap indirect ({N_BOOT_MEDIATION} resamples, percentile): "
          f"95% CI [{med_ci[0]:.4f}, {med_ci[1]:.4f}]")

    pd.DataFrame({
        "Path": ["a (PUFA->Amygdala)", "b (Amygdala->CBCL)",
                 "c' (Direct PUFA->CBCL)", "Indirect", "Sobel_SE", "Sobel_Z",
                 "Sobel_P", "Bootstrap_Indirect_Mean", "Bootstrap_CI_Lower",
                 "Bootstrap_CI_Upper", "N"],
        "Estimate": [a, b, c_prime, indirect, sobel_se, z_sobel, p_sobel,
                     boots.mean(), med_ci[0], med_ci[1], len(df_med)],
    }).to_csv(out("results_mediation.csv"), index=False)

    # -------------------------------------------------------------------------
    banner("PRIMARY ANALYSES COMPLETE")
    print(f"\nResults written to: {results_dir}")
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".csv"):
            print(f"  {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="HMinds PUFA primary analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="Path to the de-identified analysis CSV")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS,
                    help="Directory for output CSVs")
    args = ap.parse_args()
    main(args.data, args.results_dir)
