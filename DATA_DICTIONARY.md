# Expected columns

`main_analyses.py` reads a single de-identified CSV with one row per mother–infant
dyad. Only the columns below are used; any others are ignored. Missing values are
left blank and handled by multiple imputation where the analysis calls for it.

No identifier column is required or used. Do not include names, dates of birth,
record numbers, scan dates, or free text.

## Predictor

| Column | Type | Description |
|---|---|---|
| `nb_fa_poly` | numeric | Total maternal PUFA intake, g/day, from 24-h dietary recall |

## Covariates

| Column | Type | Description |
|---|---|---|
| `InfantSex` | `"F"` / `"M"` | Infant sex at birth; recoded to `InfantSex_num` (0 = female, 1 = male) |
| `MomAgeEnroll` | numeric | Maternal age at enrollment, years |
| `PrenatalBMI` | numeric | Maternal pregestational body mass index, kg/m² |
| `BirthWeight` | numeric | Infant birth weight, grams |
| `PMA` | numeric | Postmenstrual age at MRI scan, weeks (gestational age at birth + chronological age) |
| `TBV` | numeric | Total brain volume, mm³ (volumetric models only) |
| `meanFD` | numeric | Mean framewise displacement, mm (connectivity models only) |
| `nb_energy_kcal` | numeric | Total energy intake, kcal/day |
| `epds_score` | numeric | Edinburgh Postnatal Depression Scale total |
| `BirthType` | `"Vaginal"` / `"C-section"` / `"Forceps"` | Mode of delivery; recoded to `DeliveryNonVaginal` (0 = vaginal, 1 = non-vaginal) |
| `SES_binary` | `"High"` / `"Low"` | Collapsed Brazilian Economic Classification Criteria (CCEB) class: A1/B1/B2/C1 = High, C2/DE = Low; recoded to `SES_bin` (0 = high/middle, 1 = low) |
| `SES` | string | Full CCEB class, descriptives only |
| `GestAgeRecall` | numeric | Gestational age at dietary recall, weeks; descriptives only |

## Structural outcomes

Subcortical volumes in mm³:

`LeftHippocampus`, `RightHippocampus`, `LeftAmygdala`, `RightAmygdala`,
`LeftCaudate`, `RightCaudate`, `LeftPutamen`, `RightPutamen`

## Connectivity outcomes

Thirty-two within-hemisphere seed–target Fisher z-transformed correlations, four
subcortical seeds × four ipsilateral prefrontal targets per hemisphere. Hippocampus
and amygdala pairs use camelCase, caudate and putamen pairs use dotted names; both
spellings are inherited from the connectivity toolbox output and are listed verbatim
in `FC_OUTCOMES` in `main_analyses.py`. Examples:

`LHip_LLatOrb`, `LAmy_LRosAntCing`, `RCaud_R.med.orb`, `RPut_R.lat.orb`

Targets are lateral orbitofrontal cortex (`LatOrb` / `lat.orb`), medial orbitofrontal
cortex (`MedOrb` / `med.orb`), pars orbitalis (`ParOrb` / `par.orb`), and rostral
anterior cingulate cortex (`RosAntCing` / `ros.ant.cing`).

## Behavioral outcomes

Child Behavior Checklist for Ages 1½–5 T scores, one column per scale, named
`cbcl_<scale>_s`. The script selects them by that pattern, so any number of scales
will be picked up. Two are named explicitly for the bootstrap and mediation:

`cbcl_stress_prob_s`, `cbcl_anxious_depres_s`

**Naming caution:** the `_s` suffix holds T scores, which is what every analysis uses.
If the extract also carries a `_t` suffix, that holds raw item totals. The convention
is counterintuitive; check before reusing this code on a different extract.
