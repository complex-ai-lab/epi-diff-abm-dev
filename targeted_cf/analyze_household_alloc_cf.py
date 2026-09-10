"""
Analyse a household-level vs individual-level allocation CF run (CF-E, plan
sec 10.2) with per-iteration uncertainty.

Sibling of targeted_cf/analyze_age_priority_cf.py, restricted to the
household-allocation family (CF 7 reference + 25..28). Same discipline:

  * daily-case line plots       -> mean +/- 10-90 percentile band across iters
  * age-group bar charts        -> bar = mean, error bar = +/- 1 SD across iters
  * delta-vs-CF7 bars / HEADLINE -> PER-ITERATION PAIRED difference (CRN pairs
                                    iteration i across CF types), mean +/- SD

Plus one CF-E-specific figure, `household_alloc_gap_vs_total_<FIPS>.png`
(plan sec 10.2: "report the paired delta protection-gap AND the total-infection
delta side by side"), built from spillover_summary_<FIPS>.csv when the
household-spillover pass has run (else skipped with a note).

Env:
  TAG     default household_alloc
  FIPSES  default "01031"          comma-separated

Per county reads   results/<FIPS>/TargetedCF/<TAG>/:
  counterfactuals/cf_raw_<ct>.csv                               n_iter x n_days daily cases (x k)
  counterfactuals/counterfactual_data<ct>_age_incidence_raw.csv T x (1+3*n_iter) child/adult/elderly NEW infections per iter (pre-k)
  spillover_summary_<FIPS>.csv                                  (optional) paired d_protection_gap vs CF7
  calibration/calibrated_params.txt                            26 weekly R + rate + k

Per county writes  results/<FIPS>/TargetedCF/<TAG>/:
  cf_raw_mean_<ct>.csv, cf_mean_summary.csv
  cf7_fit.png, cf_all_mean.png, R_weekly.png
  counterfactuals/counterfactual_results<ct>.png   (overwrites execute()'s single draw)
  age_breakdown_<FIPS>.csv, age_bars_<FIPS>.png
  household_alloc_gap_vs_total_<FIPS>.png          (when spillover_summary_<FIPS>.csv exists)

Roll-ups  results/TargetedCF_run/<TAG>/:
  cf_mean_summary_ALL.csv, cf7_fit_ALL.csv, age_breakdown_ALL.csv,
  age_bars_ALL.png, household_alloc_gap_vs_total_ALL.png, HEADLINE.txt

Fails loud (RuntimeError / non-zero exit) on any missing file, wrong-shape series,
empty CF, iter-count mismatch between cf_raw and the age file, or a group-sum vs
reported-cases mismatch outside [0.80, 1.03].
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = os.environ.get("TAG", "household_alloc").strip()
FIPSES = [f.strip() for f in os.environ.get("FIPSES", "01031").split(",") if f.strip()]
ROLLUP_DIR = os.path.join("results", "TargetedCF_run", TAG)

CF_NAMES = {
    7:  ("07_timevar_factual_school_factual_work", "FACTUAL (individual)"),
    25: ("25_hhalloc_random",                      "hh-alloc random"),
    26: ("26_hhalloc_elderly_home_first",          "hh-alloc elderly-home"),
    27: ("27_hhalloc_largest_first",               "hh-alloc largest-first"),
    28: ("28_hhalloc_essential_home_first",        "hh-alloc ess-home"),
}
CF_ORDER = [7, 25, 26, 27, 28]
CF7_REF = 7
GROUPS = ["child", "adult", "elderly"]

# Palette: matplotlib tab10 in its natural order, indexed by CF position -- same
# convention as analyze_age_priority_cf.py, so the FACTUAL reference (CF7) is
# blue, then CF25 orange / CF26 green / CF27 red / CF28 purple. CF25-28 land on
# the same colours analyze_spillover.py gives them, so the two plot families
# agree.
_TAB10 = list(plt.cm.tab10.colors)
PAL = {ct: _TAB10[i % 10] for i, ct in enumerate(CF_ORDER)}


def die(msg):
    raise RuntimeError(msg)


def county_dir(fips):
    return os.path.join("results", fips, "TargetedCF", TAG)


def cf_dir(fips):
    return os.path.join(county_dir(fips), "counterfactuals")


def load_raw(fips, ct):
    """n_iter x n_days daily cases (x k)."""
    base = cf_dir(fips)
    cands = [
        os.path.join(base, f"cf_raw_{ct}.csv"),
        os.path.join(base, "all_counterfactual_results", CF_NAMES[ct][0], f"cf_raw_{ct}.csv"),
    ]
    for c in cands:
        if os.path.exists(c):
            r = np.loadtxt(c, delimiter=",")
            r = r if r.ndim == 2 else r[None, :]
            if r.shape[0] == 0 or r.shape[1] == 0:
                die(f"{c}: empty cf_raw for cf={ct} fips={fips}")
            return r
    die(f"cf_raw_{ct}.csv not found for fips={fips} (looked in {cands})")


def actual_series(fips):
    d = os.path.join(cf_dir(fips), "all_counterfactual_results",
                     CF_NAMES[7][0], "data", f"{fips}_counterfactual_data7.csv")
    if not os.path.exists(d):
        die(f"actual-cases file not found: {d}")
    df = pd.read_csv(d)
    if "actual_cases" not in df.columns:
        die(f"{d}: no 'actual_cases' column")
    return df["actual_cases"].to_numpy(dtype=float)


def load_age_incidence_per_iter(fips, ct, n_days_eval, n_iter_expected):
    """{group: np.array (n_iter,)} of NEW infections summed over the trailing
    n_days_eval days, one value per iteration."""
    p = os.path.join(cf_dir(fips), f"counterfactual_data{ct}_age_incidence_raw.csv")
    if not os.path.exists(p):
        die(f"per-iter age-incidence file not found: {p}")
    df = pd.read_csv(p).sort_values("t")
    if df.isna().any().any():
        die(f"{p}: contains NaN")
    if len(df) < n_days_eval:
        die(f"{p}: only {len(df)} rows, need >= {n_days_eval}")
    tail = df.iloc[len(df) - n_days_eval:]

    per = {g: {} for g in GROUPS}
    for col in df.columns:
        if col == "t":
            continue
        mobj = re.fullmatch(r"(child|adult|elderly)_new_(\d+)", col)
        if not mobj:
            die(f"{p}: unexpected column {col!r}")
        per[mobj.group(1)][int(mobj.group(2))] = float(tail[col].sum())

    out = {}
    for g in GROUPS:
        idx = sorted(per[g])
        if idx != list(range(len(idx))):
            die(f"{p}: {g} iterations not 0..N contiguous: {idx}")
        out[g] = np.array([per[g][i] for i in idx], dtype=float)
    n = len(out["child"])
    if not (len(out["adult"]) == len(out["elderly"]) == n):
        die(f"{p}: per-group iteration counts differ")
    if n != n_iter_expected:
        die(f"{p}: {n} iterations but cf_raw_{ct}.csv has {n_iter_expected}")
    return out


def calibrated_R_and_k(fips):
    p = os.path.join(county_dir(fips), "calibration", "calibrated_params.txt")
    if not os.path.exists(p):
        die(f"calibrated params not found: {p}")
    v = np.loadtxt(p).ravel()
    if v.size < 3:
        die(f"{p}: expected >= 3 params, got {v.size}")
    return v[:26], float(v[-1])          # weekly R (26), k


def regimes(a, s):
    def rmse(lo, hi):
        return float(np.sqrt(np.mean((a[lo:hi] - s[lo:hi]) ** 2)))
    return dict(rmse_peak_10_30=rmse(10, 30), rmse_plateau_35_95=rmse(35, 95),
                mae_tail_130=float(np.mean(np.abs(a[130:] - s[130:]))),
                rmse_all_14=rmse(14, len(a)),
                logrmse_all_14=float(np.sqrt(np.mean(
                    (np.log1p(a[14:]) - np.log1p(s[14:])) ** 2))),
                sim_peak=float(s.max()), actual_peak=float(a.max()))


def _daily_cf_plot(day, actual, m, lo, hi, title, path_list):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(day, actual, color="gray", lw=2.2, alpha=0.6, label="actual")
    ax.plot(day, m, color="#1f77b4", lw=1.8, label="CF mean (n-iter)")
    ax.fill_between(day, lo, hi, color="#1f77b4", alpha=0.20, label="10-90 percentile")
    ax.set_title(title); ax.set_xlabel("simulation day"); ax.set_ylabel("daily cases")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for p in path_list:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=150)
    plt.close(fig)


def load_spillover_gap(fips):
    """{ct: (d_protection_gap_mean_pp, d_protection_gap_sd_pp)} from
    spillover_summary_<FIPS>.csv (values stored as fractions -> x100 for pp).
    Returns None if the file is absent (the spillover pass has not run)."""
    p = os.path.join(county_dir(fips), f"spillover_summary_{fips}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).set_index("cf")
    out = {}
    for ct in CF_ORDER[1:]:
        if ct not in df.index:
            continue
        if "d_protection_gap_mean" not in df.columns:
            return None
        out[ct] = (float(df.loc[ct]["d_protection_gap_mean"]) * 100.0,
                   float(df.loc[ct].get("d_protection_gap_sd", np.nan)) * 100.0)
    return out or None


def _gap_vs_total_panel(fips, age_df, gap, title, path):
    """Left: paired delta total new infections vs CF7 (absolute, +/- 1 SD).
    Right: paired delta household protection gap vs CF7 (pp, +/- 1 SD)."""
    d = age_df.set_index("cf")
    non_ref = CF_ORDER[1:]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(non_ref))
    tot_m = [d.loc[ct]["d_total_mean"] for ct in non_ref]
    tot_s = [d.loc[ct]["d_total_sd"] for ct in non_ref]
    for xi, ct in zip(x, non_ref):
        axes[0].bar([xi], [d.loc[ct]["d_total_mean"]], 0.6,
                    yerr=[d.loc[ct]["d_total_sd"]], capsize=3,
                    color=PAL[ct], error_kw=dict(lw=0.9),
                    label=f"{ct} {CF_NAMES[ct][1]}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(c) for c in non_ref])
    axes[0].set_xlabel("CF type")
    axes[0].set_ylabel("Δ total new infections vs CF7 (paired)")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)

    if gap is not None:
        gcts = [c for c in non_ref if c in gap]
        xg = np.arange(len(gcts))
        for xi, ct in zip(xg, gcts):
            m, s = gap[ct]
            axes[1].bar([xi], [m], 0.6, yerr=[s if np.isfinite(s) else 0.0],
                        capsize=3, color=PAL[ct],
                        error_kw=dict(lw=0.9), label=f"{ct} {CF_NAMES[ct][1]}")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xticks(xg); axes[1].set_xticklabels([str(c) for c in gcts])
        axes[1].set_xlabel("CF type")
        axes[1].set_ylabel("Δ household protection gap vs CF7 (pp, paired)")
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "spillover_summary_<FIPS>.csv not found\n"
                     "(run the household-spillover pass)", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9)
        axes[1].set_axis_off()

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_county(fips):
    cdir = county_dir(fips)
    if not os.path.isdir(cf_dir(fips)):
        die(f"{fips}: no counterfactuals dir under {cdir}")

    actual = actual_series(fips)
    n_days = len(actual)
    R, k = calibrated_R_and_k(fips)
    if not (0.15 < k <= 1.0):
        die(f"{fips}: calibrated k={k} outside plausible (0.2, 1]")

    means, summ_rows, age_rows = {}, [], []
    age_iter = {}                       # ct -> {group: np.array (n_iter,)}
    cf7_tot_per_iter = None
    for ct in CF_ORDER:
        r = load_raw(fips, ct)
        if r.shape[1] != n_days:
            die(f"{fips} cf={ct}: cf_raw has {r.shape[1]} days, actual has {n_days}")
        n_iter = r.shape[0]
        m = r.mean(0)
        lo, hi = np.percentile(r, 10, 0), np.percentile(r, 90, 0)
        means[ct] = (m, lo, hi, n_iter)
        np.savetxt(os.path.join(cdir, f"cf_raw_mean_{ct}.csv"), m, fmt="%.4f")

        tot_per_iter = r.sum(1)
        peak_per_iter = r.max(1)
        if ct == 7:
            cf7_tot_per_iter = tot_per_iter
        summ_rows.append(dict(
            fips=fips, cf=ct, policy=CF_NAMES[ct][1], n_iter=n_iter,
            peak_mean=round(float(peak_per_iter.mean()), 1),
            peak_sd=round(float(peak_per_iter.std()), 1),
            last_mean=round(float(m[-1]), 1),
            total_mean=int(round(tot_per_iter.mean())),
            total_sd=int(round(tot_per_iter.std())),
            total_p10=int(round(np.percentile(tot_per_iter, 10))),
            total_p90=int(round(np.percentile(tot_per_iter, 90))),
        ))

        ai = load_age_incidence_per_iter(fips, ct, n_days, n_iter)
        age_iter[ct] = ai
        grp_sum_mean = sum(ai[g].mean() for g in GROUPS)
        reported_unscaled = float(m.sum()) / k
        ratio = grp_sum_mean / reported_unscaled if reported_unscaled > 0 else 0.0
        if not (0.80 <= ratio <= 1.03):
            die(f"{fips} cf={ct}: age-group new-infections sum ({grp_sum_mean:.0f}) vs "
                f"reported cases / k ({reported_unscaled:.0f}) ratio {ratio:.3f} "
                f"outside [0.80, 1.03] -- age masks or incidence logging is wrong")
        row = dict(fips=fips, cf=ct, policy=CF_NAMES[ct][1], n_iter=n_iter,
                   groupsum_vs_reported_ratio=round(ratio, 3))
        for g in GROUPS:
            row[f"{g}_cases_mean"] = int(round(ai[g].mean()))
            row[f"{g}_cases_sd"] = int(round(ai[g].std()))
        tot_iter = ai["child"] + ai["adult"] + ai["elderly"]
        row["total_new_infections_mean"] = int(round(tot_iter.mean()))
        row["total_new_infections_sd"] = int(round(tot_iter.std()))
        age_rows.append(row)

    # --- paired (per-iteration) deltas vs CF7 --------------------------------
    ref = age_iter[7]
    for row in age_rows:
        ct = row["cf"]
        for g in GROUPS + ["total"]:
            if g == "total":
                cur = age_iter[ct]["child"] + age_iter[ct]["adult"] + age_iter[ct]["elderly"]
                base = ref["child"] + ref["adult"] + ref["elderly"]
            else:
                cur, base = age_iter[ct][g], ref[g]
            d = cur - base
            row[f"d_{g}_mean"] = int(round(d.mean()))
            row[f"d_{g}_sd"] = int(round(d.std()))
            row[f"d_{g}_p10"] = int(round(np.percentile(d, 10)))
            row[f"d_{g}_p90"] = int(round(np.percentile(d, 90)))
            with np.errstate(divide="ignore", invalid="ignore"):
                dpct = np.where(base != 0, (cur - base) / base * 100.0, np.nan)
            row[f"d_{g}_pct_mean"] = round(float(np.nanmean(dpct)), 2)
            row[f"d_{g}_pct_sd"] = round(float(np.nanstd(dpct)), 2)

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(cdir, "cf_mean_summary.csv"), index=False)
    age_df = pd.DataFrame(age_rows)
    age_df.to_csv(os.path.join(cdir, f"age_breakdown_{fips}.csv"), index=False)

    day = np.arange(n_days)

    for ct in CF_ORDER:
        m, lo, hi, ni = means[ct]
        title = (f"{fips}  {TAG}  CF{ct}  {CF_NAMES[ct][1]}\n"
                 f"daily cases: {ni}-iter mean +/- 10-90 percentile band vs actual")
        paths = [
            os.path.join(cf_dir(fips), f"counterfactual_results{ct}.png"),
            os.path.join(cf_dir(fips), "all_counterfactual_results", CF_NAMES[ct][0],
                         "graphs", f"{fips}_counterfactual_results{ct}.png"),
        ]
        _daily_cf_plot(day, actual, m, lo, hi, title, paths)

    m7, lo7, hi7, ni7 = means[7]
    _daily_cf_plot(day, actual, m7, lo7, hi7,
                   f"{fips}  {TAG}  CF7 (individual alloc) vs actual -- {ni7}-iter mean +/- 10-90 band",
                   [os.path.join(cdir, "cf7_fit.png")])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(day, actual, color="gray", lw=2.0, alpha=0.5, label="actual")
    for ct in CF_ORDER:
        m, lo, hi, _ = means[ct]
        ax.fill_between(day, lo, hi, color=PAL[ct], alpha=0.10)
        ax.plot(day, m, color=PAL[ct], lw=1.5, label=f"{ct} {CF_NAMES[ct][1]}")
    ax.set_title(f"{fips}  {TAG}  household-allocation CFs -- mean + 10-90 band (daily cases)")
    ax.set_xlabel("simulation day"); ax.set_ylabel("daily cases")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(cdir, "cf_all_mean.png"), dpi=150); plt.close(fig)

    nwk = len(R); edges = np.arange(nwk + 1) * 7
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(edges, np.append(R, R[-1]), where="post", color="#1f77b4", lw=1.8)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(0, n_days)
    ax.set_title(f"{fips}  {TAG}  weekly R (reused log1p_rw2_l5_rmax5 calibration)")
    ax.set_xlabel("simulation day"); ax.set_ylabel("R"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(cdir, "R_weekly.png"), dpi=150); plt.close(fig)

    _age_bar_panels(age_df, age_iter, f"{fips}  {TAG}", os.path.join(cdir, f"age_bars_{fips}.png"))

    gap = load_spillover_gap(fips)
    _gap_vs_total_panel(fips, age_df, gap, f"{fips}  {TAG}  gap vs total (paired vs CF7)",
                        os.path.join(cdir, f"household_alloc_gap_vs_total_{fips}.png"))
    if gap is None:
        print(f"[note] {fips}: spillover_summary_{fips}.csv absent -> "
              f"gap-vs-total right panel is a placeholder (run the hSAR pass, then re-analyse)")

    fit = {"fips": fips, "tag": TAG, **regimes(actual, m7),
           "actual_total": int(actual.sum()), "cf7_total": int(m7.sum()),
           "cf7_total_sd": int(round(cf7_tot_per_iter.std())),
           "k": round(k, 4), "n_iter": ni7}
    print(f"\n===== {fips}  ({TAG}) =====")
    print(summ.to_string(index=False))
    print(age_df[["cf", "policy", "child_cases_mean", "adult_cases_mean",
                  "elderly_cases_mean", "d_total_mean", "d_total_sd"]].to_string(index=False))
    return summ, fit, age_df, age_iter


def _bar_with_err(ax, cts, get_mean, get_sd, xoff0=-3, w=0.15):
    x = np.arange(len(GROUPS))
    for i, ct in enumerate(cts):
        ax.bar(x + (i + xoff0) * w, [get_mean(ct, g) for g in GROUPS], w,
               yerr=[get_sd(ct, g) for g in GROUPS], capsize=2, color=PAL[ct],
               error_kw=dict(lw=0.8), label=f"{ct} {CF_NAMES[ct][1]}")
    ax.set_xticks(x); ax.set_xticklabels(GROUPS); ax.grid(alpha=0.3, axis="y")


def _age_bar_panels(age_df, age_iter, title, path):
    d = age_df.set_index("cf")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    _bar_with_err(axes[0], CF_ORDER,
                  lambda ct, g: d.loc[ct][f"{g}_cases_mean"],
                  lambda ct, g: d.loc[ct][f"{g}_cases_sd"], xoff0=-2)
    axes[0].set_ylabel("new infections")
    axes[0].legend(fontsize=7, ncol=2)

    _bar_with_err(axes[1], CF_ORDER[1:],
                  lambda ct, g: d.loc[ct][f"d_{g}_pct_mean"],
                  lambda ct, g: d.loc[ct][f"d_{g}_pct_sd"], xoff0=-1.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("new infection vs CF7 (%)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(ROLLUP_DIR, exist_ok=True)
    all_summ, all_fit, all_age = [], [], []
    per_fips_gap = {}
    for f in FIPSES:
        s, fit, a, ai = run_county(f)
        all_summ.append(s); all_fit.append(fit); all_age.append(a)
        per_fips_gap[f] = load_spillover_gap(f)

    pd.concat(all_summ).to_csv(os.path.join(ROLLUP_DIR, "cf_mean_summary_ALL.csv"), index=False)
    pd.DataFrame(all_fit).to_csv(os.path.join(ROLLUP_DIR, "cf7_fit_ALL.csv"), index=False)
    age_all = pd.concat(all_age)
    age_all.to_csv(os.path.join(ROLLUP_DIR, "age_breakdown_ALL.csv"), index=False)

    fig, axes = plt.subplots(1, len(FIPSES), figsize=(7 * len(FIPSES), 5.5), squeeze=False)
    for j, f in enumerate(FIPSES):
        d = age_all[age_all.fips == f].set_index("cf")
        _bar_with_err(axes[0][j], CF_ORDER[1:],
                      lambda ct, g: d.loc[ct][f"d_{g}_mean"],
                      lambda ct, g: d.loc[ct][f"d_{g}_sd"], xoff0=-1.5)
        axes[0][j].axhline(0, color="k", lw=0.8)
        axes[0][j].set_title(f"{f}  paired delta vs CF7 (+/- 1 SD)")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(ROLLUP_DIR, "age_bars_ALL.png"), dpi=150); plt.close(fig)

    # --- gap-vs-total roll-up: row 0 = Δ total infections, row 1 = Δ protection
    #     gap; one column per county; per-CF bars in the shared palette ---------
    non_ref = CF_ORDER[1:]
    x = np.arange(len(non_ref))
    fig, axes = plt.subplots(2, len(FIPSES), figsize=(5.2 * len(FIPSES), 8.4),
                             squeeze=False)
    for j, f in enumerate(FIPSES):
        d = age_all[age_all.fips == f].set_index("cf")
        gap = per_fips_gap[f]
        for xi, ct in zip(x, non_ref):
            axes[0][j].bar([xi], [d.loc[ct]["d_total_mean"]], 0.62,
                           yerr=[d.loc[ct]["d_total_sd"]], capsize=3,
                           color=PAL[ct], error_kw=dict(lw=0.9),
                           label=f"{ct} {CF_NAMES[ct][1]}")
        axes[0][j].axhline(0, color="k", lw=0.8)
        axes[0][j].set_xticks(x); axes[0][j].set_xticklabels([str(c) for c in non_ref])
        axes[0][j].set_ylabel("Δ total new infections vs CF7")
        axes[0][j].set_title(f)
        axes[0][j].grid(alpha=0.3, axis="y")

        if gap is not None:
            for xi, ct in zip(x, non_ref):
                if ct not in gap:
                    continue
                m, s = gap[ct]
                axes[1][j].bar([xi], [m], 0.62, yerr=[s if np.isfinite(s) else 0.0],
                               capsize=3, color=PAL[ct], error_kw=dict(lw=0.9))
            axes[1][j].axhline(0, color="k", lw=0.8)
            axes[1][j].set_xticks(x); axes[1][j].set_xticklabels([str(c) for c in non_ref])
            axes[1][j].set_ylabel("Δ household protection gap vs CF7 (pp)")
            axes[1][j].grid(alpha=0.3, axis="y")
        else:
            axes[1][j].text(0.5, 0.5, "spillover_summary_<FIPS>.csv not found",
                            ha="center", va="center", transform=axes[1][j].transAxes,
                            fontsize=9)
            axes[1][j].set_axis_off()
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{TAG}  gap vs total (paired vs CF7)  --  top: Δ total infections, "
                 f"bottom: Δ household protection gap", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROLLUP_DIR, "household_alloc_gap_vs_total_ALL.png"), dpi=150)
    plt.close(fig)

    lines = ["household-level allocation CF (CF-E, plan sec 10.2) -- per-iteration PAIRED delta",
             "in NEW infections vs CF7 (individual, uniform-over-susceptibles allocation).",
             "CRN pairs iteration i across CF types, so d_i = total(CFk, i) - total(CF7, i);",
             "reported as mean +/- SD  [p10, p90] over the iterations. Single seed (42).",
             "Runs on covid_output_causal + reused log1p_rw2_l5_rmax5 (no essnet, no recal).",
             "See spillover HEADLINE (results/TargetedCF_run/spillover/) for the paired",
             "delta household protection gap -- the other half of the CF-E question.",
             f"TAG={TAG}   counties={','.join(FIPSES)}", ""]
    for f in FIPSES:
        d = age_all[age_all.fips == f].set_index("cf")
        n = int(d.loc[7]["n_iter"])
        r7 = d.loc[7]
        lines.append(f"[{f}]  n_iter={n}   CF7 totals (mean +/- SD): "
                     f"child={r7.child_cases_mean}+/-{r7.child_cases_sd}  "
                     f"adult={r7.adult_cases_mean}+/-{r7.adult_cases_sd}  "
                     f"elderly={r7.elderly_cases_mean}+/-{r7.elderly_cases_sd}  "
                     f"all={r7.total_new_infections_mean}+/-{r7.total_new_infections_sd}")
        gap = per_fips_gap[f]
        for ct in CF_ORDER[1:]:
            row = d.loc[ct]
            gtxt = ""
            if gap is not None and ct in gap:
                gtxt = "   d_protection_gap %+.2f +/- %.2f pp" % gap[ct]
            lines.append("   CF%-2d %-22s%s" % (ct, CF_NAMES[ct][1], gtxt))
            for g in ["child", "adult", "elderly", "total"]:
                lines.append("        d_%-8s %+7d +/- %5d   [%+7d, %+7d]"
                             % (g, row[f"d_{g}_mean"], row[f"d_{g}_sd"],
                                row[f"d_{g}_p10"], row[f"d_{g}_p90"]))
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(ROLLUP_DIR, "HEADLINE.txt"), "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"[saved] {ROLLUP_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[FAIL] {e}\n", file=sys.stderr)
        sys.exit(1)
