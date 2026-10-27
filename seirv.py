"""Discrete-time SEIRV model: the ABM with MIXING: mean_field and STAGE_DURATIONS: exponential,
without agents.

    .venv/bin/python seirv.py                                   # parameters from config.yaml
    .venv/bin/python seirv.py --iters 30 --out seirv_model.csv  # + stochastic (chain binomial) runs
    .venv/bin/python seirv.py --r-t-csv results/01031/mf/r_t_daily.csv --compare results/01031/mf

One step per day, in the ABM's order. With S, E, I, R, V at the end of day t-1 and N agents:

    p     = 1 - exp(-R_t * I / N)          infection probability of each susceptible
    v     = min(doses(t), S)               S -> V, doses from intervention.csv
    new_E = (S - v) * p                    S -> E
    E -> I: E * (1 - exp(-1 / EXPOSED_TO_INFECTED_TIME))
    I -> R: I * (1 - exp(-1 / INFECTED_TO_RECOVERED_TIME))
    R -> S: R * (1 - exp(-RECOVERED_WANING_RATE)),  V -> S: V * (1 - exp(-VACCINATED_WANING_RATE))
            (IMMUNITY_WANING_MODE: stochastic; none turns waning off)

The INITIAL_INFECTED fraction starts in I on day 0 and cannot recover before day 1. R_t per day
follows the config like run.sh does (R_T_BURN_IN in the burn-in, then R_T or R_T_DAILY), or comes
from --r-t-csv. `cases` is K x S x p, which is what the ABM reports: it counts exposures before
same-day vaccination removes some of them, so it is slightly more than K x new_E.

The expected-value columns are the deterministic recursion above. With --iters N the same steps
run with binomial draws N times, which is what the agent-based mean-field model does; *_mean and
*_sd are over those runs. --compare prints how far a run.sh output (seirv_daily.csv) is from both.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))


def r_t_per_day(epi, burn_in, num_steps, r_t_csv=None):
    """R_t for simulation days 0..num_steps-1, the way the simulator picks it."""
    if r_t_csv:
        df = pd.read_csv(r_t_csv).set_index("day")["r_t"]
        missing = [d for d in range(num_steps) if d not in df.index]
        if missing:
            raise ValueError(f"{r_t_csv} has no R_t for days {missing[:5]}...")
        return df.reindex(range(num_steps)).to_numpy(float)

    r_t, daily = epi.get("R_T"), epi.get("R_T_DAILY")
    if r_t is None and daily is None:
        raise ValueError("set R_T or R_T_DAILY in the config, or pass --r-t-csv "
                         "(the r_t_daily.csv of a run that used a calibration)")
    out = np.full(num_steps, np.nan)
    if r_t is not None:
        weekly = np.atleast_1d(np.asarray(r_t, dtype=float))
        num_weeks = 26
        if weekly.size == 1:
            weekly = np.repeat(weekly, num_weeks)
        if weekly.size != num_weeks:
            raise ValueError(f"R_T must be one number or {num_weeks} weekly values, got {weekly.size}")
        for t in range(num_steps):
            out[t] = weekly[min(max((t - burn_in) // 7, 0), num_weeks - 1)]
    if epi.get("R_T_BURN_IN") is not None:
        out[:burn_in] = float(epi["R_T_BURN_IN"])
    if daily is not None:
        if isinstance(daily, str):
            path = daily if os.path.isabs(daily) else os.path.join(ROOT, daily)
            df = pd.read_csv(path)
            if "day" in df.columns:
                df = df.sort_values("day")
                start, vals = int(df["day"].iloc[0]), df["r_t"].to_numpy(float)
            else:
                start, vals = burn_in, df["r_t"].to_numpy(float)
        else:
            start, vals = burn_in, np.asarray(list(daily), dtype=float)
        if start > burn_in or start + len(vals) < num_steps:
            raise ValueError(f"R_T_DAILY covers days {start}..{start + len(vals) - 1}, "
                             f"the run needs {burn_in}..{num_steps - 1}")
        out[start:start + len(vals)] = vals[:num_steps - start]
    if np.isnan(out).any():
        raise ValueError("R_t is undefined on some days (R_T_DAILY without R_T and R_T_BURN_IN?)")
    return out


def daily_doses(population_dir, num_steps):
    """Doses per day as the factual run gives them: int(65+ doses) + int(other doses)."""
    iv = pd.read_csv(os.path.join(ROOT, population_dir, "intervention.csv")).set_index("t")
    iv = iv.reindex(range(num_steps))
    if "vaccines_65plus" in iv.columns:
        v65 = iv["vaccines_65plus"].to_numpy(float)
        rest = np.maximum(iv["vaccines"].to_numpy(float) - v65, 0.0)
        return np.floor(v65) + np.floor(rest), iv["date"].to_numpy()
    return np.floor(iv["vaccines"].to_numpy(float)), iv["date"].to_numpy()


def simulate(n, i0, r_t, doses, p_e, p_i, p_r, p_v, k, rng=None):
    """One trajectory. rng=None: expected values; otherwise binomial draws."""
    draw = (lambda m, p: m * p) if rng is None else (lambda m, p: float(rng.binomial(int(round(m)), p)))
    seeded = n * i0 if rng is None else float(rng.binomial(int(n), i0))
    s, e, i, r, v = n - seeded, 0.0, seeded, 0.0, 0.0
    rows = []
    for t in range(len(r_t)):
        p = 1.0 - math.exp(-r_t[t] * i / n)
        vac = min(doses[t], s)
        new_e = draw(s - vac, p)
        e_i = draw(e, p_e)
        i_r = draw(i, p_i) if t > 0 else 0.0    # seeded agents leave I from day 1
        r_s = draw(r, p_r)
        v_s = draw(v, p_v)
        cases = k * s * p
        s, e, i, r, v = (s - vac - new_e + r_s + v_s, e + new_e - e_i, i + e_i - i_r,
                         r + i_r - r_s, v + vac - v_s)
        rows.append((t, s, e, i, r, v, new_e, cases))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--r-t-csv", help="R_t per simulation day (a run's r_t_daily.csv); overrides the config")
    ap.add_argument("--i0", type=float, help="initial infected fraction (default INITIAL_INFECTED)")
    ap.add_argument("--k", type=float, help="reporting fraction for `cases` (default K, else 1)")
    ap.add_argument("--no-vaccines", action="store_true", help="no S -> V (like CF 11)")
    ap.add_argument("--iters", type=int, default=0, help="stochastic runs to add (default 0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="seirv_model.csv")
    ap.add_argument("--compare", help="run.sh output folder to compare against (its seirv_daily.csv)")
    a = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(a.config), resolve=True)
    epi, md = cfg["epidemic"], cfg["simulation_metadata"]
    burn_in = int(md.get("BURN_IN_DAYS", 0) or 0)
    num_steps = burn_in + int(epi.get("HORIZON_DAYS") or md["num_steps_per_episode"] - burn_in)
    n = float(md["num_agents"])
    i0 = a.i0 if a.i0 is not None else epi.get("INITIAL_INFECTED")
    if i0 is None:
        raise SystemExit("INITIAL_INFECTED is null in the config: pass --i0 (row 27 of calibrated_params.txt)")
    k = a.k if a.k is not None else (epi.get("K") if epi.get("K") is not None else 1.0)

    waning = epi.get("IMMUNITY_WANING_MODE", "none")
    if waning == "fixed":
        raise SystemExit("IMMUNITY_WANING_MODE: fixed has no SEIRV equivalent; use none or stochastic")
    p_r = 1 - math.exp(-epi["RECOVERED_WANING_RATE"]) if waning == "stochastic" else 0.0
    p_v = 1 - math.exp(-epi["VACCINATED_WANING_RATE"]) if waning == "stochastic" else 0.0
    p_e = 1 - math.exp(-1 / epi["EXPOSED_TO_INFECTED_TIME"])
    p_i = 1 - math.exp(-1 / epi["INFECTED_TO_RECOVERED_TIME"])
    if (epi.get("MIXING", "network"), epi.get("STAGE_DURATIONS", "fixed")) != ("mean_field", "exponential"):
        print("[note] this config is not MIXING: mean_field + STAGE_DURATIONS: exponential, so the ABM "
              "run with it is not the SEIRV model")

    r_t = r_t_per_day(epi, burn_in, num_steps, a.r_t_csv)
    doses, dates = daily_doses(md["population_dir"], num_steps)
    if a.no_vaccines or not md.get("WITH_VACC", True):
        doses = np.zeros(num_steps)

    cols = ["S", "E", "I", "R", "V", "new_E", "cases"]
    exp_traj = simulate(n, i0, r_t, doses, p_e, p_i, p_r, p_v, k)
    out = pd.DataFrame(exp_traj[:, 1:], columns=cols)
    out.insert(0, "phase", np.where(np.arange(num_steps) < burn_in, "burn_in", "observed"))
    out.insert(0, "date", dates)
    out.insert(0, "day", np.arange(num_steps))
    out.insert(3, "r_t", r_t)
    if a.iters > 0:
        rng = np.random.default_rng(a.seed)
        runs = np.stack([simulate(n, i0, r_t, doses, p_e, p_i, p_r, p_v, k, rng)[:, 1:] for _ in range(a.iters)])
        for j, c in enumerate(cols):
            out[f"{c}_mean"] = runs[:, :, j].mean(0)
            out[f"{c}_sd"] = runs[:, :, j].std(0)
    out.to_csv(a.out, index=False)
    print(f"N {n:.0f}, I0 {i0:.6f}, K {k:.4f}, {num_steps} days; daily p(E->I) {p_e:.4f}, p(I->R) {p_i:.4f}, "
          f"waning {waning}; wrote {a.out}")

    if a.compare:
        abm = pd.read_csv(os.path.join(a.compare, "seirv_daily.csv")).set_index("day").reindex(out["day"])
        print(f"ABM ({a.compare}) minus SEIRV, as % of N (max over days):")
        for c in ["S", "E", "I", "R", "V"]:
            line = f"  {c}: expected-value {100 * np.nanmax(np.abs(abm[c].values - out[c].values)) / n:.3f}%"
            if a.iters > 0:
                line += (f", stochastic mean {100 * np.nanmax(np.abs(abm[c].values - out[c + '_mean'].values)) / n:.3f}%"
                         f" (its SD up to {100 * out[c + '_sd'].max() / n:.3f}%)")
            print(line)


if __name__ == "__main__":
    main()
