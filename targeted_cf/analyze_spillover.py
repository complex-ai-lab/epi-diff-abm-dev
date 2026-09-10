"""
Analyzer for the household-spillover hSAR outputs written by spillover_hook.py.

Reads, per county per family, the per-CF wide files
  results/<FIPS>/TargetedCF/<TAG>/counterfactuals/counterfactual_data<ct>_hh_spillover_raw.csv
and produces:
  spillover_summary_<FIPS>.csv     hSAR + paired-delta table (CF vs CF7)
  spillover_hSAR_<FIPS>.png        14-day hSAR time series, mean + 10-90 band
  spillover_bars_<FIPS>.png        left: absolute hSAR by CF; right: paired
                                   delta protection-gap vs CF7 (percentage points)
  spillover_age_bars_<FIPS>.png    child/adult/elderly: paired delta hSAR vs CF7
Cross-county roll-up (results/TargetedCF_run/spillover/):
  spillover_summary_ALL.csv, spillover_bars_ALL.png, spillover_age_bars_ALL.png,
  HEADLINE.txt

Style matches analyze_age_priority_cf.py / analyze_essentiality_cf.py: Agg
backend, no plot titles, shared tab10 palette across CF types, bar = mean /
whisker = +/- 1 SD across iterations, every "CFk vs CF7" number is the
per-iteration PAIRED delta (CRN pairs iteration i), reported mean +/- SD
[p10, p90].

Fails loud on a missing/short raw file, a day-index mismatch between CF types,
or a NaN in the paired delta of the overall vacc / unvacc hSAR (structurally-NaN
cuts -- the 1-person-household columns -- are reported as n/a, not raised).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# hSAR is undefined (NaN) on every pre-vaccine-rollout day for the vaccinated-
# household series; nanmean/nanpercentile over an all-NaN warm-up slice is
# expected, not an error.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

WINDOW = 14
CF7_REF = 7

CF_NAMES = {
    7:  "FACTUAL (uniform)",
    12: "elderly-1st schF/wkF",
    13: "workage-1st schF/wkF",
    14: "workage-1st schC/wkO",
    15: "workage-1st schO/wkO",
    16: "elderly-1st schO/wkC",
    17: "elderly-1st schC/wkC",
    18: "essfirst occ-factual",
    19: "noness-close uniform",
    20: "noness-close essfirst",
    21: "essfirst all-open",
    22: "noness-close uniform schC",
    23: "noness-close essfirst schC",
    25: "hh-alloc random",
    26: "hh-alloc elderly-home",
    27: "hh-alloc largest-first",
    28: "hh-alloc ess-home",
}
AGE_CUTS = ["child", "adult", "elderly"]
SIZE_CUTS = ["hh2", "hh3p"]        # hh1 = singletons, structurally NaN, reported separately

ROLLUP_DIR = os.path.join("results", "TargetedCF_run", "spillover")


def die(msg):
    raise RuntimeError(msg)


def _palette(cf_types):
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    out = {}
    for i, ct in enumerate(cf_types):
        out[ct] = (0.55, 0.55, 0.55, 1.0) if ct == CF7_REF else tuple(cmap[i % 10])
    return out


def _rollout_day(fips):
    p = os.path.join("populations", f"pop{fips}", "intervention.csv")
    if not os.path.isfile(p):
        return None
    iv = pd.read_csv(p)
    nz = iv[iv["vaccines"] > 0]
    return int(nz["t"].iloc[0]) if len(nz) else None


# --------------------------------------------------------------------------
def _load_raw(cf_dir, ct):
    p = os.path.join(cf_dir, f"counterfactual_data{ct}_hh_spillover_raw.csv")
    if not os.path.isfile(p):
        die(f"spillover raw file missing: {p}")
    df = pd.read_csv(p)
    if len(df) < 100:
        die(f"{p}: only {len(df)} day rows")
    return df


def _metric_matrix(df, metric):
    """[n_days, n_iter] for one metric name (columns '<metric>_<i>')."""
    cols = sorted([c for c in df.columns if c.startswith(metric + "_")
                   and c[len(metric) + 1:].isdigit()],
                  key=lambda c: int(c[len(metric) + 1:]))
    if not cols:
        die(f"metric {metric!r} not found in spillover raw file")
    return df[cols].to_numpy(dtype=float), df["t"].to_numpy(dtype=int)


def _per_iter_mean(df, metric, day_lo):
    """day-averaged (nanmean over days >= day_lo) per iteration -> [n_iter].

    day_lo is the vaccine-rollout day (not WINDOW): hSAR_vacc is undefined before
    any household has a vaccinated member, so integrating hSAR_unvacc from day 14
    while hSAR_vacc only exists post-rollout makes the protection gap a non-like-
    for-like comparison and inflates it for CF policies that shift the epidemic
    earlier (e.g. CF14/15 in 36003). Restricting BOTH series to post-rollout days
    fixes that; the full pre-rollout hSAR dynamics stay visible in
    spillover_hSAR_<FIPS>.png."""
    mat, t = _metric_matrix(df, metric)
    sel = t >= day_lo
    with np.errstate(invalid="ignore"):
        return np.nanmean(mat[sel], axis=0)


def _stat(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(np.nanmean(a)), sd=float(np.nanstd(a)),
                p10=float(np.nanpercentile(a, 10)), p90=float(np.nanpercentile(a, 90)))


def _paired(cur, ref, allow_nan, label):
    d = cur - ref
    if not allow_nan and np.isnan(d).any():
        die(f"paired delta for {label} has {int(np.isnan(d).sum())} NaN of {d.size} "
            f"iterations -- a whole summary window was empty; investigate before trusting the plot")
    return d


# --------------------------------------------------------------------------
def analyze_one(fips, cf_types, output_dir, family):
    cf_dir = os.path.join(output_dir, "counterfactuals")
    raw = {ct: _load_raw(cf_dir, ct) for ct in cf_types}

    t0 = raw[CF7_REF]["t"].to_numpy()
    for ct, df in raw.items():
        if not np.array_equal(df["t"].to_numpy(), t0):
            die(f"cf={ct}: day index differs from CF7")

    roll = _rollout_day(fips)
    day_lo = max(WINDOW, roll if roll is not None else WINDOW)

    # per-iteration day-averaged values (post-rollout window; see _per_iter_mean)
    piv = {}          # piv[ct][metric] -> [n_iter]
    for ct in cf_types:
        piv[ct] = {}
        for metric in (["hSAR_vacc", "hSAR_unvacc"]
                       + [f"hSAR_vacc_{g}" for g in AGE_CUTS + SIZE_CUTS + ["hh1"]]
                       + [f"hSAR_unvacc_{g}" for g in AGE_CUTS + SIZE_CUTS + ["hh1"]]):
            piv[ct][metric] = _per_iter_mean(raw[ct], metric, day_lo)

    ref = piv[CF7_REF]
    rows = []
    for ct in cf_types:
        cur = piv[ct]
        row = {"fips": fips, "family": family, "cf": ct, "policy": CF_NAMES.get(ct, str(ct)),
               "n_iter": len(cur["hSAR_vacc"]), "summary_day_lo": day_lo}
        for side in ("vacc", "unvacc"):
            for k, v in _stat(cur[f"hSAR_{side}"]).items():
                row[f"hSAR_{side}_{k}"] = round(v, 5)
            if ct != CF7_REF:
                d = _paired(cur[f"hSAR_{side}"], ref[f"hSAR_{side}"], allow_nan=False,
                            label=f"{fips} cf{ct} hSAR_{side}")
                for k, v in _stat(d).items():
                    row[f"dSAR_{side}_{k}"] = round(v, 5)

        gap = cur["hSAR_unvacc"] - cur["hSAR_vacc"]
        gref = ref["hSAR_unvacc"] - ref["hSAR_vacc"]
        row["protection_gap_mean"] = round(float(np.nanmean(gap)), 5)
        row["protection_gap_sd"] = round(float(np.nanstd(gap)), 5)
        if ct != CF7_REF:
            dgap = _paired(gap, gref, allow_nan=False, label=f"{fips} cf{ct} protection_gap")
            for k, v in _stat(dgap).items():
                row[f"d_protection_gap_{k}"] = round(v, 5)

        for g in AGE_CUTS + SIZE_CUTS:
            gp = cur[f"hSAR_unvacc_{g}"] - cur[f"hSAR_vacc_{g}"]
            row[f"{g}_protection_gap_mean"] = round(float(np.nanmean(gp)), 5)
            if ct != CF7_REF:
                gpr = ref[f"hSAR_unvacc_{g}"] - ref[f"hSAR_vacc_{g}"]
                dgp = _paired(gp, gpr, allow_nan=True, label=f"{fips} cf{ct} {g} gap")
                row[f"{g}_d_protection_gap_mean"] = round(float(np.nanmean(dgp)), 5)
                row[f"{g}_d_protection_gap_sd"] = round(float(np.nanstd(dgp)), 5)
                for side in ("vacc", "unvacc"):
                    dd = _paired(cur[f"hSAR_{side}_{g}"], ref[f"hSAR_{side}_{g}"],
                                 allow_nan=True, label=f"{fips} cf{ct} {g} {side}")
                    row[f"{g}_dSAR_{side}_mean"] = round(float(np.nanmean(dd)), 5)
                    row[f"{g}_dSAR_{side}_sd"] = round(float(np.nanstd(dd)), 5)
        rows.append(row)

    summ = pd.DataFrame(rows)
    summ_path = os.path.join(output_dir, f"spillover_summary_{fips}.csv")
    summ.to_csv(summ_path, index=False)

    _plot_hsar_timeseries(fips, cf_types, raw, family, output_dir)
    _plot_bars(fips, cf_types, summ, piv, ref, family, output_dir)
    _plot_age_bars(fips, cf_types, summ, family, output_dir)
    print(f"[saved] {summ_path} + 3 plots", flush=True)
    return summ


# --------------------------------------------------------------------------
def _plot_hsar_timeseries(fips, cf_types, raw, family, output_dir):
    pal = _palette(cf_types)
    roll = _rollout_day(fips)
    t0_days = raw[CF7_REF]["t"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, side, lab in zip(axes, ("vacc", "unvacc"),
                             ("susceptibles WITH a vaccinated cohabitant",
                              "susceptibles with NO vaccinated cohabitant")):
        for ct in cf_types:
            mat, t = _metric_matrix(raw[ct], f"hSAR_{side}")
            with np.errstate(invalid="ignore"):
                m = np.nanmean(mat, axis=1)
                lo = np.nanpercentile(mat, 10, axis=1)
                hi = np.nanpercentile(mat, 90, axis=1)
            ax.plot(t, m, color=pal[ct], lw=1.6, label=f"{ct} {CF_NAMES.get(ct, ct)}")
            ax.fill_between(t, lo, hi, color=pal[ct], alpha=0.12)
        if roll is not None:
            ax.axvline(roll, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(0, int(t0_days[-1]))
        ax.set_xlabel("simulation day")
        ax.set_ylabel("14-day household SAR")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top", fontsize=8)
    axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"spillover_hSAR_{fips}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _bar_group(ax, cats, series, colors, ylabel, width=None):
    x = np.arange(len(cats))
    n = len(series)
    w = width or 0.8 / max(n, 1)
    for i, (lab, vals, errs) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, yerr=errs, capsize=2,
               color=colors[i], error_kw=dict(lw=0.8), label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")


def _plot_bars(fips, cf_types, summ, piv, ref, family, output_dir):
    pal = _palette(cf_types)
    d = summ.set_index("cf")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cats = [str(ct) for ct in cf_types]
    vacc_m = [d.loc[ct]["hSAR_vacc_mean"] for ct in cf_types]
    vacc_s = [d.loc[ct]["hSAR_vacc_sd"] for ct in cf_types]
    unv_m = [d.loc[ct]["hSAR_unvacc_mean"] for ct in cf_types]
    unv_s = [d.loc[ct]["hSAR_unvacc_sd"] for ct in cf_types]
    _bar_group(axes[0], cats,
               [("vaccinated-hh susceptibles", vacc_m, vacc_s),
                ("unvaccinated-hh susceptibles", unv_m, unv_s)],
               ["#4c78a8", "#e45756"], "mean 14-day household SAR")
    axes[0].set_xlabel("CF type")
    axes[0].legend(fontsize=8)

    non_ref = [ct for ct in cf_types if ct != CF7_REF]
    x = np.arange(len(non_ref))
    dg_m = [d.loc[ct]["d_protection_gap_mean"] for ct in non_ref]
    dg_s = [d.loc[ct]["d_protection_gap_sd"] for ct in non_ref]
    for xi, ct in zip(x, non_ref):
        axes[1].bar([xi], [d.loc[ct]["d_protection_gap_mean"] * 100.0], 0.6,
                    yerr=[d.loc[ct]["d_protection_gap_sd"] * 100.0], capsize=3,
                    color=pal[ct], error_kw=dict(lw=0.9), label=f"{ct}  {CF_NAMES.get(ct, ct)}")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(ct) for ct in non_ref])
    axes[1].set_xlabel("CF type")
    axes[1].set_ylabel("Δ protection gap vs CF7 (percentage points)")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"spillover_bars_{fips}.png"), dpi=150)
    plt.close(fig)


def _plot_age_bars(fips, cf_types, summ, family, output_dir):
    pal = _palette(cf_types)
    d = summ.set_index("cf")
    non_ref = [ct for ct in cf_types if ct != CF7_REF]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(AGE_CUTS))
    w = 0.8 / max(len(non_ref), 1)
    for i, ct in enumerate(non_ref):
        m = [d.loc[ct][f"{g}_d_protection_gap_mean"] * 100.0 for g in AGE_CUTS]
        s = [d.loc[ct][f"{g}_d_protection_gap_sd"] * 100.0 for g in AGE_CUTS]
        ax.bar(x + (i - (len(non_ref) - 1) / 2) * w, m, w, yerr=s, capsize=2,
               color=pal[ct], error_kw=dict(lw=0.8), label=f"{ct} {CF_NAMES.get(ct, ct)}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_CUTS)
    ax.set_ylabel("Δ protection gap vs CF7 (pp) — susceptible cohabitant age group")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"spillover_age_bars_{fips}.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def _rollup(all_summ, families, fipses):
    os.makedirs(ROLLUP_DIR, exist_ok=True)
    cat = pd.concat(all_summ, ignore_index=True)
    cat.to_csv(os.path.join(ROLLUP_DIR, "spillover_summary_ALL.csv"), index=False)

    for family in families:
        sub = cat[cat.family == family]
        if sub.empty:
            continue
        fam_fips = [f for f in fipses if not sub[sub.fips == f].empty]
        cf_types = sorted(sub.cf.unique())
        non_ref = [c for c in cf_types if c != CF7_REF]

        fig, axes = plt.subplots(1, len(fam_fips), figsize=(6 * len(fam_fips), 4.6),
                                 squeeze=False)
        for j, f in enumerate(fam_fips):
            df = sub[sub.fips == f].set_index("cf")
            x = np.arange(len(non_ref))
            for xi, ct in zip(x, non_ref):
                axes[0][j].bar([xi], [df.loc[ct]["d_protection_gap_mean"] * 100.0], 0.6,
                               yerr=[df.loc[ct]["d_protection_gap_sd"] * 100.0], capsize=3,
                               color=_palette(cf_types)[ct], error_kw=dict(lw=0.9),
                               label=f"{ct}  {CF_NAMES.get(ct, ct)}")
            axes[0][j].axhline(0, color="k", lw=0.8)
            axes[0][j].set_xticks(x)
            axes[0][j].set_xticklabels([str(ct) for ct in non_ref])
            axes[0][j].set_xlabel("CF type")
            axes[0][j].set_ylabel("Δ protection gap vs CF7 (pp)")
            axes[0][j].text(0.02, 0.97, f, transform=axes[0][j].transAxes, va="top", fontsize=9)
            axes[0][j].grid(alpha=0.3, axis="y")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROLLUP_DIR, f"spillover_bars_ALL_{family}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(fam_fips), figsize=(6 * len(fam_fips), 4.6),
                                 squeeze=False, sharey=True)
        for j, f in enumerate(fam_fips):
            df = sub[sub.fips == f].set_index("cf")
            x = np.arange(len(AGE_CUTS))
            w = 0.8 / max(len(non_ref), 1)
            for i, ct in enumerate(non_ref):
                mm = [df.loc[ct][f"{g}_d_protection_gap_mean"] * 100.0 for g in AGE_CUTS]
                ss = [df.loc[ct][f"{g}_d_protection_gap_sd"] * 100.0 for g in AGE_CUTS]
                axes[0][j].bar(x + (i - (len(non_ref) - 1) / 2) * w, mm, w, yerr=ss, capsize=2,
                               color=_palette(cf_types)[ct], error_kw=dict(lw=0.7),
                               label=f"{ct} {CF_NAMES.get(ct, ct)}")
            axes[0][j].axhline(0, color="k", lw=0.8)
            axes[0][j].set_xticks(x)
            axes[0][j].set_xticklabels(AGE_CUTS)
            axes[0][j].set_ylabel("Δ protection gap vs CF7 (pp)")
            axes[0][j].text(0.02, 0.97, f, transform=axes[0][j].transAxes, va="top", fontsize=9)
            axes[0][j].grid(alpha=0.3, axis="y")
        axes[0][-1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join(ROLLUP_DIR, f"spillover_age_bars_ALL_{family}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    _headline(cat, families, fipses)


def _headline(cat, families, fipses):
    L = ["=== Household Spillover Analysis (hSAR, 14-day rolling window) ===",
         "Metric: household secondary attack rate for susceptible cohabitants.",
         "protection gap = hSAR(no vaccinated cohabitant) - hSAR(>=1 vaccinated cohabitant).",
         "All 'CFk vs CF7' numbers are per-iteration PAIRED deltas (CRN pairs iter i),",
         "reported mean +/- SD [p10, p90], in percentage points. Single seed (42).",
         "1-person households have no cohabitant -> excluded from every hSAR (hh1 = n/a).",
         "Summary integrates hSAR over days >= the county's vaccine-rollout day only",
         "(hSAR_vacc is undefined before any household is vaccinated; a day-14 start",
         "would count the pre-rollout wave on the unvacc side only and inflate the gap).",
         "The 'elderly' age cut under elderly-first CFs (12/16/17) is a tiny-N cell",
         "(few elderly are both still-susceptible AND living with a vaccinated member)",
         "-- treat its large +pp values as small-denominator noise, not signal.",
         ""]
    for family in families:
        sub = cat[cat.family == family]
        if sub.empty:
            continue
        L.append(f"---- family: {family} ----")
        for f in fipses:
            df = sub[sub.fips == f]
            if df.empty:
                continue
            df = df.set_index("cf")
            reg = os.path.join("results", str(f), "TargetedCF", family,
                               f"spillover_cf7_regression_{f}.txt")
            reg_line = ""
            if os.path.isfile(reg):
                reg_line = "  ".join(open(reg).read().split("\n")[2:5])
            L.append(f"[{f}]  n_iter={int(df.loc[CF7_REF]['n_iter'])}   {reg_line}")
            r7 = df.loc[CF7_REF]
            L.append(f"   CF7  hSAR vacc-hh {r7['hSAR_vacc_mean']:.4f}+/-{r7['hSAR_vacc_sd']:.4f}"
                     f"   unvacc-hh {r7['hSAR_unvacc_mean']:.4f}+/-{r7['hSAR_unvacc_sd']:.4f}"
                     f"   gap {r7['protection_gap_mean']*100:+.2f} pp")
            for ct in [c for c in df.index if c != CF7_REF]:
                row = df.loc[ct]
                L.append(f"   CF{ct:<2d} {CF_NAMES.get(ct, ''):<24s} "
                         f"d_gap {row['d_protection_gap_mean']*100:+.2f} +/- "
                         f"{row['d_protection_gap_sd']*100:.2f}  "
                         f"[{row['d_protection_gap_p10']*100:+.2f}, {row['d_protection_gap_p90']*100:+.2f}] pp")
                L.append("        by age  " + "  ".join(
                    f"{g}:{row[f'{g}_d_protection_gap_mean']*100:+.2f}" for g in AGE_CUTS)
                    + "   by hh-size  " + "  ".join(
                        f"{g}:{row[f'{g}_d_protection_gap_mean']*100:+.2f}" for g in SIZE_CUTS))
            L.append("")
    L.append("Cross-family note: age-priority spillover uses covid_output_causal + "
             "log1p_rw2_l5_rmax5; essentiality uses covid_output_essnet + (possibly) "
             "..._essnet. hSAR magnitudes are NOT directly comparable across families.")
    txt = "\n".join(L)
    with open(os.path.join(ROLLUP_DIR, "HEADLINE.txt"), "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fipses", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["age_priority"])
    ap.add_argument("--cf-types", default=None,
                    help="comma list; default depends on family")
    args = ap.parse_args()

    fam_cf = {"age_priority": [7, 12, 13, 14, 15, 16, 17],
              "household_alloc": [7, 25, 26, 27, 28],
              "essentiality": [7, 18, 19, 20, 21, 22, 23]}
    all_summ = []
    for family in args.families:
        cf_types = ([int(x) for x in args.cf_types.split(",")] if args.cf_types
                    else fam_cf[family])
        for f in args.fipses:
            out = os.path.join("results", str(f), "TargetedCF", family)
            if not os.path.isdir(os.path.join(out, "counterfactuals")):
                print(f"[skip] {f} {family}: no counterfactuals dir")
                continue
            all_summ.append(analyze_one(f, cf_types, out, family))
    if all_summ:
        _rollup(all_summ, args.families, [str(f) for f in args.fipses])


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[FAIL] {e}\n", file=sys.stderr)
        sys.exit(1)
