# Runbook: `log1p_rw2_l5` calibration + counterfactuals for a new waning config

**Purpose.** Re-run the current-best pipeline (`log1p` loss + RW2 λ5 + endpoint
anchor λ5 + `PARAM_SNAPSHOT_K=20` + fixed intervention-severity stochasticity +
`R_MAX=2.5`) for 17117 / 01031 / 36003 under a different immunity-waning config,
and produce weekly R + calibration + counterfactual artifacts + comparison
roll-ups. Use this whenever the ask is "redo the multi-county run but with
waning = X".

Background / rationale for every knob:
`COVID-ABM/Discussion/Regularization_discussion.md` §5 and
`results/LossCompare_run/INTERVENTION_STOCHASTICITY.md`.

---

## 1. What to run

All scripts live in `results/LossCompare_run/`. Env: `.venv_ds`
(`/home/pkbiswas/.conda/envs/.venv_ds/bin/python`), A100, SEED 42.

### One command (all 3 counties + analysis)

```bash
cd /home/pkbiswas/COVID-ABM/epi-diff-abm-dev
RTS=90 VTS=135 TAG=log1p_rw2_l5_fixed90 \
  bash results/LossCompare_run/run_all_waning_variant.sh
```

- `RTS` = `RECOVERED_TO_SUSCEPTIBLE_TIME` (days, fixed waning mode)
- `VTS` = `VACCINATED_TO_SUSCEPTIBLE_TIME` (days). Convention: `VTS = 1.5 * RTS`.
- `TAG` = output subfolder name. Convention: `log1p_rw2_l5_fixed<RTS>`.
- Optional: `FIPSES="17117 01031 36003"`, `SEED=42`.

Runtime ≈ 30–35 min/county (calibration 251 epochs + 11 CF × 30 iters) →
**~1.5–2 h total**. Run in the background and monitor the per-county logs under
`results/LossCompare_run/<TAG>/logs/`.

### Per-county only

```bash
FIPS=17117 RTS=90 VTS=135 TAG=log1p_rw2_l5_fixed90 \
  bash results/LossCompare_run/run_waning_variant.sh
```

### Analysis / plots only (after the runs exist)

```bash
FIPSES=17117,01031,36003 TAG=log1p_rw2_l5_fixed90 \
  /home/pkbiswas/.conda/envs/.venv_ds/bin/python \
  results/LossCompare_run/analyze_waning_variant.py
```

### Defaults reproduce the committed baseline

With no `RTS/VTS/TAG` overrides the scripts reproduce `run_log1p_rw2_l5.sh`
exactly: `RTS=60`, `VTS=90`, `TAG=log1p_rw2_l5`.

---

## 2. Fixed knobs (do not change without a reason — see §5 of the discussion)

| knob | value | set by |
|---|---|---|
| `LOSS_MODE` | `log1p` | driver |
| `R_SMOOTH_ORDER` | 2 (RW2 on `log R`) | driver |
| `R_SMOOTH_LAMBDA` | 5 | driver |
| `R_SMOOTH_ENDPOINT_LAMBDA` | 5 (first-diff anchor on w1/w2, w25/w26) | driver |
| `PARAM_SNAPSHOT_K` | 20 (saved params = mean of last 20 epochs) | driver |
| `R_MAX` | 2.5 (weekly-R box upper bound; was 5.0) | driver, overridable |
| epochs | 251 | hardcoded `abm_nets.py:448` |
| CF types | all 11 | default |
| CF iterations | 30 | default (`abm_nets.py:390`) |
| SEED | 42 | env |
| waning mode | `fixed` | driver (`set_vacc_config.py --mode fixed`) |
| `WITH_K` / `WITH_VACC` / `USE_7DAY_AVG` | true / true / false | already in `config.yaml` |

`config.yaml` is patched in place per run and **restored on exit** (trap). The
committed baseline, `result_graphs/`, and `vaccinated_test/` are never touched.

---

## 3. What it produces & where

### Per county — `results/<FIPS>/LossCompare/<TAG>/`

| path | contents |
|---|---|
| `RUN_CONFIG.txt` | exact knobs + timestamp for this run |
| `calibration/calibrated_params.txt` | weekly R (26) + 2 rate params, mean of last 20 epochs |
| `calibration/` | full training snapshots, loss csv, `generated_factual.csv`, `_curriculum_phase1_novacc/` |
| `calibration/loss.png`, `.../<epoch>/…` | training diagnostics |
| `counterfactuals/all_counterfactual_results/<CF>/cf_raw_<ct>.csv` | 30 × 168 daily-case ensemble per CF type |
| `counterfactuals/all_counterfactual_results/<CF>/data/<FIPS>_counterfactual_data<ct>.csv` | single-draw daily cases + `actual_cases` |
| **`R_weekly.png`** | calibrated weekly R as a **stepwise / piecewise-constant** trace on the **simulation-day x-axis** (week w = days `[7(w-1), 7w)`), so it lines up 1:1 with `cf7_fit.png` / `cf_all_mean.png` for reading R against the CF trajectories; a secondary week axis is drawn on top. Written by the analysis step. |
| **`cf7_fit.png`** | CF7 factual 30-iter mean ± 10–90% band vs actual, 3-regime shading |
| **`cf_all_mean.png`** | all 11 CF means on one axis |
| `cf_raw_mean_<ct>.csv` | reduced 30-iteration mean trajectory per CF type |
| `cf_mean_summary.csv` | per CF type: peak (+p10/p90), last day, total |

### Cross-county roll-ups — `results/LossCompare_run/<TAG>/`

| path | contents |
|---|---|
| `cf_mean_summary_ALL.csv` | every county × CF type: peak / last / total |
| `cf7_fit_ALL.csv` | per county: peak/plateau/tail RMSE, log-RMSE, CF7 total vs actual, `sum_abs_dlogR`, R_mean, **CF11 peak/last/total** |
| `logs/run_<FIPS>.log`, `logs/analyze.log` | full stdout |

The three asks — **weekly R**, **calibration**, **counterfactuals** — map to
`R_weekly.png` + `calibration/calibrated_params.txt`, the `calibration/` folder,
and `counterfactuals/` + `cf_*` respectively.

---

## 4. How to read the result

1. **Factual fit** — `cf7_fit_ALL.csv`: `rmse_all_14`, `logrmse_all_14`,
   `cf7_total` vs `actual_total` (want within ~10%), `mae_tail_130` (want small,
   flat tail). Eyeball `cf7_fit.png` per county.
2. **Weekly R** — `R_weekly.png`, `sum_abs_dlogR` (roughness; RW2 λ5 typically
   ~3–4). Watch the tail weeks w20–w26.
3. **Counterfactuals** — `cf_mean_summary_ALL.csv`. Key contrasts:
   - work-closure lever: CF1/6/9/10 (open work) vs CF7 (factual).
   - **CF11 (no vaccines)** blow-up: `cf11_total` vs `cf7_total`, `cf11_last`
     vs `cf7_last`. The open question (discussion §5.7 / §4) is whether a
     longer waning (bigger `RTS/VTS`) tames CF11's tail ramp for 17117 / 36003.
4. **Compare variants** — diff `cf7_fit_ALL.csv` and `cf_mean_summary_ALL.csv`
   across `results/LossCompare_run/<TAG>/` folders. `fixed_60` baseline lives at
   `results/LossCompare_run/{cf7_fit_ALL,cf_mean_summary_ALL}.csv` (no subfolder).

---

## 5. Variant log

| TAG | RTS / VTS | notes |
|---|---|---|
| `log1p_rw2_l5` | 60 / 90 | committed baseline (`INTERVENTION_STOCHASTICITY.md` §3); roll-ups at `results/LossCompare_run/*_ALL.csv` |
| `log1p_rw2_l5_rmax5` | 60 / 90 | same but `R_MAX=5.0`; `results/LossCompare_run_rmax5/` |
| `log1p_rw2_l5_fixed90` | 90 / 135 | 2026-09-02. Longer natural + vaccine immunity. **Result: barely moves anything.** Calibrated weekly R ≈ identical to `fixed_60` (few agents wane back inside the 168-day window); CF7 fit equivalent (17117 RMSE 13.6→13.7, 36003 16.2→16.0); CF11 tail ramp for 17117 *persists / slightly worse* (last 164→181, total +76% vs CF7). Confirms discussion §5.7: the CF11 blow-up is the non-identified tail-R flat direction, not the waning rate. |

Append a row here each time this is re-run with new waning values.

---

## 6. Gotchas

- If a run dies mid-way, `config.yaml` is still restored by the trap; just
  re-invoke (outputs are `rm -rf`'d and rebuilt per county).
- The GNN scatter ops have no deterministic CUDA kernel
  (`torch.use_deterministic_algorithms(True, warn_only=True)`), so tail R and
  CF11 vary run-to-run at fixed seed. For a load-bearing CF11 number, run ≥ 3
  seeds and report the mean (discussion §5.4).
- `NETWORKS_DIR` in `.env` points at scratch
  (`/scratch/alrodri_root/alrodri/pkbiswas/epi-diff-abm-dev/networks/covid_output_causal`).
  The 3 counties here are already prepared.
- All 3 counties share the window 2020-10-26 → 2021-04-25 (26 weeks, 182 days).
