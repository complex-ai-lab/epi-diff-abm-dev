"""
Analyse a work-essentiality CF run (CF-B) with per-iteration uncertainty.

Sibling of analyze_age_priority_cf.py. GROUPS = essential / non-essential /
not-in-workforce, keyed on the ACS-C24030 patched population. Cases only.

Every mean carries a spread (same as CF-A):
  * daily-case line plots     -> mean + 10-90 percentile band across iters
  * group bar charts          -> bar = mean, whisker = +/- 1 SD across iters
  * delta-vs-CF7 / HEADLINE    -> per-iteration PAIRED difference, mean +/- SD [p10,p90]

Extra roll-up: targeted_vs_full_closure_ALL.csv
  retained_benefit_i = (CF7_total_i - CF19_total_i) / (CF7_total_i - CF24_total_i)
  = fraction of the full-blanket-closure transmission drop that a
    non-essential-only closure (CF19) still delivers. CF 24 = ALL occ closed,
    same networks + school as the rest of the run.

Env:  TAG (default essentiality)   FIPSES (default "01031")

Fails loud on any missing file, wrong-shape series, iter-count mismatch, or a
group-sum vs reported-cases ratio outside [0.80, 1.03].
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = os.environ.get("TAG", "essentiality").strip()
FIPSES = [f.strip() for f in os.environ.get("FIPSES", "01031").split(",") if f.strip()]
ROLLUP_DIR = os.path.join("results", "TargetedCF_run", TAG)

CF_NAMES = {
    7:  ("07_timevar_factual_school_factual_work",         "FACTUAL (uniform)"),
    18: ("18_ess_essfirst_occfactual_schF",                "essfirst-vacc  occ-factual"),
    19: ("19_ess_nonessclose_uniformvacc_schF",            "noness-close   uniform"),
    20: ("20_ess_essfirst_nonessclose_schF",               "noness-close   essfirst"),
    21: ("21_ess_essfirst_allopen",                        "essfirst-vacc  all-open"),
    22: ("22_ess_nonessclose_uniformvacc_schC",            "noness-close   uniform schC"),
    23: ("23_ess_essfirst_nonessclose_schC",               "noness-close   essfirst schC"),
    24: ("24_ess_allocc_closed_uniformvacc_schF",          "ALL-occ-close (ref)"),
}
CF_ORDER = [7, 18, 19, 20, 21, 22, 23]     # 24 is the retained-benefit reference, not plotted in the grid
CF_REF_FULLCLOSE = 24
GROUPS = ["essential", "nonessential", "other"]
CMAP = plt.cm.tab10(np.linspace(0, 1, len(CF_ORDER)))

ESSENTIAL_SECTORS = {
    "AGRICULTURE", "MINING", "UTILITIES", "TRANSPORTATION", "HEALTHCARE", "FOOD",
    "MANUFACTURING", "CONSTRUCTION", "WASTEMANAGEMENT", "OTHER", "EDUCATION",
}
NON_ESSENTIAL_SECTORS = {
    "WHOLESALETRADE", "RETAILTRADE", "INFORMATION", "FINANCEINSURANCE",
    "REALESTATERENTAL", "SCIENTIFICTECHNICAL", "ENTERPRISEMANAGEMENT", "ART",
}


def die(m):
    raise RuntimeError(m)


def county_dir(fips):
    return os.path.join("results", fips, "TargetedCF", TAG)


def cf_dir(fips):
    return os.path.join(county_dir(fips), "counterfactuals")


def load_raw(fips, ct):
    base = cf_dir(fips)
    for c in (os.path.join(base, f"cf_raw_{ct}.csv"),
              os.path.join(base, "all_counterfactual_results", CF_NAMES[ct][0], f"cf_raw_{ct}.csv")):
        if os.path.exists(c):
            r = np.loadtxt(c, delimiter=",")
            r = r if r.ndim == 2 else r[None, :]
            if r.size == 0:
                die(f"{c}: empty")
            return r
    die(f"cf_raw_{ct}.csv not found for fips={fips}")


def actual_series(fips):
    d = os.path.join(cf_dir(fips), "all_counterfactual_results",
                     CF_NAMES[7][0], "data", f"{fips}_counterfactual_data7.csv")
    if not os.path.exists(d):
        die(f"actual-cases file not found: {d}")
    return pd.read_csv(d)["actual_cases"].to_numpy(float)


def load_group_incidence_per_iter(fips, ct, n_days_eval, n_iter_expected):
    p = os.path.join(cf_dir(fips), f"counterfactual_data{ct}_essential_incidence_raw.csv")
    if not os.path.exists(p):
        die(f"per-iter essential-incidence file not found: {p}")
    df = pd.read_csv(p).sort_values("t")
    if df.isna().any().any():
        die(f"{p}: contains NaN")
    if len(df) < n_days_eval:
        die(f"{p}: only {len(df)} rows, need >= {n_days_eval}")
    tail = df.iloc[len(df) - n_days_eval:]
    per = {"essential": {}, "nonessential": {}, "other": {}}
    for col in df.columns:
        if col == "t":
            continue
        m = re.fullmatch(r"(essential|nonessential|other)_new_(\d+)", col)
        if not m:
            die(f"{p}: unexpected column {col!r}")
        per[m.group(1)][int(m.group(2))] = float(tail[col].sum())
    out = {}
    for g in GROUPS:
        idx = sorted(per[g])
        if idx != list(range(len(idx))):
            die(f"{p}: {g} iterations not 0..N contiguous: {idx}")
        out[g] = np.array([per[g][i] for i in idx], float)
    n = len(out["essential"])
    if not (len(out["nonessential"]) == len(out["other"]) == n):
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
    return v[:26], float(v[-1])


def workforce_counts(fips):
    hits = glob.glob(os.path.join("targeted_cf", "phase2", "populations", f"{fips}_population.csv"))
    if len(hits) != 1:
        die(f"patched population for {fips} not found: {hits}")
    occ = pd.read_csv(hits[0])["Occupations"]
    e = int(occ.isin(ESSENTIAL_SECTORS).sum())
    ne = int(occ.isin(NON_ESSENTIAL_SECTORS).sum())
    return e, ne, len(occ) - e - ne


def regimes(a, s):
    def rmse(lo, hi):
        return float(np.sqrt(np.mean((a[lo:hi] - s[lo:hi]) ** 2)))
    return dict(rmse_peak_10_30=rmse(10, 30), rmse_plateau_35_95=rmse(35, 95),
                mae_tail_130=float(np.mean(np.abs(a[130:] - s[130:]))),
                rmse_all_14=rmse(14, len(a)),
                logrmse_all_14=float(np.sqrt(np.mean(
                    (np.log1p(a[14:]) - np.log1p(s[14:])) ** 2))))


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


def _bar_with_err(ax, cts, get_mean, get_sd, xoff0):
    x = np.arange(len(GROUPS))
    w = 0.8 / max(len(cts), 1)
    for i, ct in enumerate(cts):
        ax.bar(x + (i + xoff0) * w, [get_mean(ct, g) for g in GROUPS], w,
               yerr=[get_sd(ct, g) for g in GROUPS], capsize=2,
               color=CMAP[CF_ORDER.index(ct)], error_kw=dict(lw=0.8),
               label=f"{ct} {CF_NAMES[ct][1]}")
    ax.set_xticks(x); ax.set_xticklabels(GROUPS); ax.grid(alpha=0.3, axis="y")


def run_county(fips):
    if not os.path.isdir(cf_dir(fips)):
        die(f"{fips}: no counterfactuals dir")
    actual = actual_series(fips)
    n_days = len(actual)
    R, k = calibrated_R_and_k(fips)
    if not (0.15 < k <= 1.0):
        die(f"{fips}: calibrated k={k} implausible")

    means, tot_iter, summ_rows, grp_rows = {}, {}, [], []
    grp_iter = {}
    for ct in CF_ORDER + [CF_REF_FULLCLOSE]:
        r = load_raw(fips, ct)
        if r.shape[1] != n_days:
            die(f"{fips} cf={ct}: {r.shape[1]} days != actual {n_days}")
        means[ct] = (r.mean(0), np.percentile(r, 10, 0), np.percentile(r, 90, 0), r.shape[0])
        tot_iter[ct] = r.sum(1)
        np.savetxt(os.path.join(county_dir(fips), f"cf_raw_mean_{ct}.csv"), r.mean(0), fmt="%.4f")
        if ct == CF_REF_FULLCLOSE:
            continue
        summ_rows.append(dict(fips=fips, cf=ct, policy=CF_NAMES[ct][1], n_iter=r.shape[0],
                              peak_mean=round(float(r.max(1).mean()), 1),
                              peak_sd=round(float(r.max(1).std()), 1),
                              total_mean=int(round(tot_iter[ct].mean())),
                              total_sd=int(round(tot_iter[ct].std())),
                              total_p10=int(round(np.percentile(tot_iter[ct], 10))),
                              total_p90=int(round(np.percentile(tot_iter[ct], 90)))))
        if ct in (18, 19, 20, 21, 22, 23) or ct == 7:
            gi = load_group_incidence_per_iter(fips, ct, n_days, r.shape[0])
            grp_iter[ct] = gi
            gs = sum(gi[g].mean() for g in GROUPS)
            ratio = gs / (float(r.mean(0).sum()) / k)
            if not (0.80 <= ratio <= 1.03):
                die(f"{fips} cf={ct}: group-sum/reported ratio {ratio:.3f} outside [0.80,1.03]")
            row = dict(fips=fips, cf=ct, policy=CF_NAMES[ct][1], n_iter=r.shape[0],
                       groupsum_vs_reported_ratio=round(ratio, 3))
            for g in GROUPS:
                row[f"{g}_cases_mean"] = int(round(gi[g].mean()))
                row[f"{g}_cases_sd"] = int(round(gi[g].std()))
            ti = gi["essential"] + gi["nonessential"] + gi["other"]
            row["total_new_infections_mean"] = int(round(ti.mean()))
            row["total_new_infections_sd"] = int(round(ti.std()))
            grp_rows.append(row)

    # paired deltas vs CF7
    ref = grp_iter[7]
    for row in grp_rows:
        ct = row["cf"]
        for g in GROUPS + ["total"]:
            if g == "total":
                cur = grp_iter[ct]["essential"] + grp_iter[ct]["nonessential"] + grp_iter[ct]["other"]
                base = ref["essential"] + ref["nonessential"] + ref["other"]
            else:
                cur, base = grp_iter[ct][g], ref[g]
            d = cur - base
            row[f"d_{g}_mean"] = int(round(d.mean()))
            row[f"d_{g}_sd"] = int(round(d.std()))
            row[f"d_{g}_p10"] = int(round(np.percentile(d, 10)))
            row[f"d_{g}_p90"] = int(round(np.percentile(d, 90)))
            # per-iteration percent change vs CF7:  (CFk - CF7) / CF7 * 100
            with np.errstate(divide="ignore", invalid="ignore"):
                dpct = np.where(base != 0, (cur - base) / base * 100.0, np.nan)
            row[f"d_{g}_pct_mean"] = round(float(np.nanmean(dpct)), 2)
            row[f"d_{g}_pct_sd"] = round(float(np.nanstd(dpct)), 2)

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(county_dir(fips), "cf_mean_summary.csv"), index=False)
    grp_df = pd.DataFrame(grp_rows)
    grp_df.to_csv(os.path.join(county_dir(fips), f"essential_breakdown_{fips}.csv"), index=False)

    day = np.arange(n_days)
    for ct in CF_ORDER + [CF_REF_FULLCLOSE]:
        m, lo, hi, ni = means[ct]
        _daily_cf_plot(day, actual, m, lo, hi,
                       f"{fips}  {TAG}  CF{ct}  {CF_NAMES[ct][1]}\n{ni}-iter mean +/- 10-90 band vs actual",
                       [os.path.join(cf_dir(fips), f"counterfactual_results{ct}.png"),
                        os.path.join(cf_dir(fips), "all_counterfactual_results", CF_NAMES[ct][0],
                                     "graphs", f"{fips}_counterfactual_results{ct}.png")])

    m7, lo7, hi7, ni7 = means[7]
    _daily_cf_plot(day, actual, m7, lo7, hi7,
                   f"{fips}  {TAG}  CF7 (uniform, blanket) vs actual -- {ni7}-iter mean +/- 10-90 band",
                   [os.path.join(county_dir(fips), "cf7_fit.png")])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(day, actual, color="gray", lw=2.0, alpha=0.5, label="actual")
    for i, ct in enumerate(CF_ORDER):
        m, lo, hi, _ = means[ct]
        ax.fill_between(day, lo, hi, color=CMAP[i], alpha=0.10)
        ax.plot(day, m, color=CMAP[i], lw=1.5, label=f"{ct} {CF_NAMES[ct][1]}")
    ax.set_title(f"{fips}  {TAG}  essentiality CFs -- mean + 10-90 band (daily cases)")
    ax.set_xlabel("simulation day"); ax.set_ylabel("daily cases")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(county_dir(fips), "cf_all_mean.png"), dpi=150); plt.close(fig)

    nwk = len(R); edges = np.arange(nwk + 1) * 7
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(edges, np.append(R, R[-1]), where="post", color="#1f77b4", lw=1.8)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(0, n_days)
    ax.set_title(f"{fips}  {TAG}  weekly R")
    ax.set_xlabel("simulation day"); ax.set_ylabel("R"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(county_dir(fips), "R_weekly.png"), dpi=150); plt.close(fig)

    d = grp_df.set_index("cf")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    _bar_with_err(axes[0], CF_ORDER, lambda ct, g: d.loc[ct][f"{g}_cases_mean"],
                  lambda ct, g: d.loc[ct][f"{g}_cases_sd"], -3)
    axes[0].set_ylabel("new infections"); axes[0].legend(fontsize=7, ncol=2)
    _bar_with_err(axes[1], CF_ORDER[1:], lambda ct, g: d.loc[ct][f"d_{g}_pct_mean"],
                  lambda ct, g: d.loc[ct][f"d_{g}_pct_sd"], -2.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("new infection vs CF7 (%)"); axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(county_dir(fips), f"essential_bars_{fips}.png"), dpi=150); plt.close(fig)

    # retained benefit: (CF7 - CF19) / (CF7 - CF24), paired per iteration.
    # Only meaningful when the FULL closure (CF24) is itself an effective lever;
    # if closing every workplace barely moves the total (household/community-
    # driven epidemic), the ratio is a small/small quotient and unstable.
    denom = tot_iter[7] - tot_iter[CF_REF_FULLCLOSE]
    fullclose_red = float((tot_iter[7].mean() - tot_iter[CF_REF_FULLCLOSE].mean()) / tot_iter[7].mean())
    reliable = fullclose_red >= 0.25 and np.all(denom > 0)
    rb = (tot_iter[7] - tot_iter[19]) / np.where(denom > 0, denom, np.nan)
    e_cnt, ne_cnt, o_cnt = workforce_counts(fips)
    rb_row = dict(fips=fips,
                  cf7_total=int(round(tot_iter[7].mean())), cf19_total=int(round(tot_iter[19].mean())),
                  cf_fullclose_total=int(round(tot_iter[CF_REF_FULLCLOSE].mean())),
                  full_closure_reduction_pct=round(100 * fullclose_red, 1),
                  retained_benefit_reliable=bool(reliable),
                  retained_benefit_mean=round(float(np.nanmean(rb)), 3),
                  retained_benefit_sd=round(float(np.nanstd(rb)), 3),
                  retained_benefit_p10=round(float(np.nanpercentile(rb, 10)), 3),
                  retained_benefit_p90=round(float(np.nanpercentile(rb, 90)), 3),
                  essential_workers=e_cnt, nonessential_workers=ne_cnt, not_in_workforce=o_cnt,
                  essential_share_of_workers=round(e_cnt / (e_cnt + ne_cnt), 3))

    fit = {"fips": fips, "tag": TAG, **regimes(actual, m7),
           "actual_total": int(actual.sum()), "cf7_total": int(m7.sum()),
           "cf7_total_sd": int(round(tot_iter[7].std())), "k": round(k, 4), "n_iter": ni7}
    print(f"\n===== {fips}  ({TAG}) =====")
    print(summ.to_string(index=False))
    print(grp_df[["cf", "policy", "essential_cases_mean", "nonessential_cases_mean",
                  "other_cases_mean", "d_total_mean", "d_total_sd"]].to_string(index=False))
    print(f"retained_benefit (CF19 vs full-close CF24) = {rb.mean():.3f} +/- {rb.std():.3f}  "
          f"| essential workers {e_cnt} ({rb_row['essential_share_of_workers']*100:.0f}% of workforce)")
    return summ, fit, grp_df, grp_iter, rb_row


def main():
    os.makedirs(ROLLUP_DIR, exist_ok=True)
    all_summ, all_fit, all_grp, all_rb = [], [], [], []
    for f in FIPSES:
        s, fit, g, gi, rb = run_county(f)
        all_summ.append(s); all_fit.append(fit); all_grp.append(g); all_rb.append(rb)

    pd.concat(all_summ).to_csv(os.path.join(ROLLUP_DIR, "cf_mean_summary_ALL.csv"), index=False)
    pd.DataFrame(all_fit).to_csv(os.path.join(ROLLUP_DIR, "cf7_fit_ALL.csv"), index=False)
    grp_all = pd.concat(all_grp)
    grp_all.to_csv(os.path.join(ROLLUP_DIR, "essential_breakdown_ALL.csv"), index=False)
    pd.DataFrame(all_rb).to_csv(os.path.join(ROLLUP_DIR, "targeted_vs_full_closure_ALL.csv"), index=False)

    fig, axes = plt.subplots(1, len(FIPSES), figsize=(7 * len(FIPSES), 5.5), squeeze=False)
    for j, f in enumerate(FIPSES):
        d = grp_all[grp_all.fips == f].set_index("cf")
        _bar_with_err(axes[0][j], CF_ORDER[1:], lambda ct, g: d.loc[ct][f"d_{g}_mean"],
                      lambda ct, g: d.loc[ct][f"d_{g}_sd"], -2.5)
        axes[0][j].axhline(0, color="k", lw=0.8)
        axes[0][j].set_title(f"{f}  paired delta vs CF7 (+/- 1 SD)")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(ROLLUP_DIR, "essential_bars_ALL.png"), dpi=150); plt.close(fig)

    lines = ["work-essentiality CF -- per-iteration PAIRED delta in NEW infections vs CF7.",
             "d_i = total(CFk,i) - total(CF7,i); reported mean +/- SD [p10, p90].",
             "retained_benefit_i = (CF7_i - CF19_i) / (CF7_i - CF24_i)  "
             "(fraction of full-blanket-closure benefit kept by exempting essential workers).",
             "Single seed (42). Repeat 43/44 for a load-bearing number.",
             f"TAG={TAG}  counties={','.join(FIPSES)}", ""]
    for f in FIPSES:
        d = grp_all[grp_all.fips == f].set_index("cf")
        rb = next(r for r in all_rb if r["fips"] == f)
        r7 = d.loc[7]
        lines.append(f"[{f}]  n_iter={int(d.loc[7]['n_iter'])}   "
                     f"essential workers {rb['essential_workers']} "
                     f"({rb['essential_share_of_workers']*100:.0f}% of workforce), "
                     f"non-essential {rb['nonessential_workers']}, not-in-workforce {rb['not_in_workforce']}")
        _rel = "" if rb["retained_benefit_reliable"] else \
            "  [UNRELIABLE: full closure only cuts %.0f%% -> ratio ill-conditioned]" % rb["full_closure_reduction_pct"]
        lines.append(f"       retained_benefit(CF19 vs full-close CF24) = "
                     f"{rb['retained_benefit_mean']:+.3f} +/- {rb['retained_benefit_sd']:.3f}  "
                     f"[{rb['retained_benefit_p10']:+.3f}, {rb['retained_benefit_p90']:+.3f}]"
                     f"  (full closure cuts {rb['full_closure_reduction_pct']:.0f}%){_rel}")
        lines.append(f"       CF7 totals (mean +/- SD): "
                     f"essential={r7.essential_cases_mean}+/-{r7.essential_cases_sd}  "
                     f"non-ess={r7.nonessential_cases_mean}+/-{r7.nonessential_cases_sd}  "
                     f"other={r7.other_cases_mean}+/-{r7.other_cases_sd}  "
                     f"all={r7.total_new_infections_mean}+/-{r7.total_new_infections_sd}")
        for ct in CF_ORDER[1:]:
            row = d.loc[ct]
            lines.append("   CF%-2d %-28s" % (ct, CF_NAMES[ct][1]))
            for g in ["essential", "nonessential", "other", "total"]:
                lines.append("        d_%-12s %+7d +/- %5d   [%+7d, %+7d]"
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
