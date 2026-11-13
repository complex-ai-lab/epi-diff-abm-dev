"""Custom counterfactual (CF type 99) read from one YAML file.

Set `CF_TYPES: "99"` and `CF_SPEC: counterfactuals/my_cf.yaml` in config.yaml's
run_settings (or `CF_TYPES=99 CF_SPEC=... bash run.sh`). The spec lists, by date,
what is open or closed and who is vaccinated first; see
counterfactuals/TEMPLATE.yaml for every option. Days the spec does not mention
are FACTUAL, so an empty spec reproduces CF 7 exactly.

This module only turns the spec into one row per simulated day (`table`);
transition.py applies the row, and abm_nets.py writes the table as
counterfactual_interventions.csv.
"""
import datetime as _dt
import json
import os
import re

import pandas as pd
import yaml

CUSTOM_CF_TYPE = 99

# NPI lever values. Numbers are accepted as well: 0 = open, 1 = closed.
LEVER_VALUES = ("open", "closed", "factual", "flipped")
# Work-only shorthand: non-essential workplaces closed, essential ones factual (= CF 19's work lever).
WORK_SHORTHANDS = {"nonessential_closed": {"essential": "factual", "nonessential": "closed"}}

# Named vaccine priority modes (transition.py VACC_MODES).
NAMED_PRIORITIES = (
    "factual",                  # observed age-phased doses (65+ quota from vaccines_65plus, rest to under-65)
    "uniform",                  # uniform at random over susceptibles
    "none",                     # no doses
    "elderly_first",            # 65A > 50t64 > 40t49 > 30t39 > 20t29 > U19
    "working_age_first",        # 20-64 > 65A > U19
    "essential_first",          # essential-sector workers first, then uniform
    "hh_random",                # whole households, random household order
    "hh_elderly_home_first",    # whole households, those with a 65+ member first
    "hh_largest_first",         # whole households, largest first
    "hh_essential_home_first",  # whole households, those with an essential worker first
)
# Keys of a custom priority dict -> (kind, code). Age codes as in populations/*/mapping.json;
# work groups as in transition.py _require_agent_essential (0 not working, 1 essential, 2 non-essential).
AGE_KEYS = {"20t29": 0, "30t39": 1, "40t49": 2, "50t64": 3, "65A": 4, "U19": 5}
WORK_KEYS = {"not_working": 0, "essential": 1, "nonessential": 2}
TOP_KEYS = {"name", "school", "work", "vaccine"}
VACCINE_KEYS = {"doses", "schedule"}


def _date(x, where):
    if isinstance(x, _dt.date):
        return pd.Timestamp(x)
    try:
        return pd.Timestamp(str(x))
    except Exception as e:
        raise ValueError(f"CF_SPEC {where}: bad date {x!r}") from e


def _lever(v, where):
    """Normalise one lever value to 'open' / 'closed' / 'factual' / 'flipped'."""
    if isinstance(v, bool):
        raise ValueError(f"CF_SPEC {where}: use open/closed/factual/flipped or 0/1, not {v!r}")
    if v in (0, 1):
        return "closed" if v == 1 else "open"
    if isinstance(v, str) and v.strip().lower() in LEVER_VALUES:
        return v.strip().lower()
    raise ValueError(f"CF_SPEC {where}: {v!r} is not one of {LEVER_VALUES} or 0/1")


def _work(v, where):
    """Work value -> {'essential': lever, 'nonessential': lever} or a single lever (all workplaces)."""
    if isinstance(v, str) and v.strip().lower() in WORK_SHORTHANDS:
        return dict(WORK_SHORTHANDS[v.strip().lower()])
    if isinstance(v, dict):
        bad = set(v) - {"essential", "nonessential"}
        if bad or len(v) != 2:
            raise ValueError(f"CF_SPEC {where}: a split work value needs exactly the keys "
                             f"'essential' and 'nonessential', got {sorted(v)}")
        return {k: _lever(v[k], f"{where}.{k}") for k in ("essential", "nonessential")}
    return _lever(v, where)


def _priority(v, where):
    """Priority -> a named mode (str) or {"ranks": {key: rank}} for a custom ordering."""
    if isinstance(v, str):
        if v not in NAMED_PRIORITIES:
            raise ValueError(f"CF_SPEC {where}: unknown priority {v!r}; named modes are "
                             f"{NAMED_PRIORITIES}, or give a dict of ranks")
        return v
    if isinstance(v, dict):
        ranks = {}
        for k, r in v.items():
            if not isinstance(k, str) or k not in AGE_KEYS and k not in WORK_KEYS:
                raise ValueError(f"CF_SPEC {where}: priority key {k!r} must be a string, one of "
                                 f"{list(AGE_KEYS)} (age bands) or {list(WORK_KEYS)} (work groups)")
            if isinstance(r, bool) or not isinstance(r, (int, float)) or r < 0:
                raise ValueError(f"CF_SPEC {where}: rank for {k!r} must be a number >= 0, got {r!r}")
            ranks[k] = float(r)
        if not ranks:
            raise ValueError(f"CF_SPEC {where}: empty priority dict")
        return {"ranks": ranks}
    raise ValueError(f"CF_SPEC {where}: priority must be a mode name or a dict, got {v!r}")


def _expand(value, dates, default, parse, where, item_key="value"):
    """A whole-run value or a list of {from, to, <item_key>} ranges -> one value per date.
    Ranges are applied in order (a later range overrides an earlier one); `to` is
    inclusive and defaults to the last day; uncovered days keep `default`."""
    out = [default] * len(dates)
    if value is None:
        return out
    if not isinstance(value, list):
        v = parse(value, where)
        return [v] * len(dates)
    first, last = dates[0], dates[-1]
    for i, item in enumerate(value):
        w = f"{where}[{i}]"
        if not isinstance(item, dict) or item_key not in item or "from" not in item:
            raise ValueError(f"CF_SPEC {w}: each range needs 'from' and '{item_key}' "
                             f"(and optionally 'to'), got {item!r}")
        bad = set(item) - {"from", "to", item_key}
        if bad:
            raise ValueError(f"CF_SPEC {w}: unknown keys {sorted(bad)}")
        lo = _date(item["from"], w)
        hi = _date(item["to"], w) if item.get("to") is not None else last
        if hi < lo:
            raise ValueError(f"CF_SPEC {w}: 'to' {hi.date()} is before 'from' {lo.date()}")
        if lo > last or hi < first:
            raise ValueError(f"CF_SPEC {w}: {lo.date()}..{hi.date()} is outside the simulated "
                             f"days {first.date()}..{last.date()}")
        v = parse(item[item_key], w)
        for j, d in enumerate(dates):
            if lo <= d <= hi:
                out[j] = v
    return out


def spec_name(spec, spec_path):
    """The spec's `name:` (default: the file name). It names the output folder, so it must be folder-safe."""
    name = str(spec.get("name") or os.path.splitext(os.path.basename(spec_path))[0])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"CF_SPEC {spec_path}: name {name!r} names the output folder; "
                         f"use only letters, digits, _ . -")
    return name


def _resolve(lever, factual):
    return {"open": 0, "closed": 1, "factual": int(factual), "flipped": 1 - int(factual)}[lever]


def _label(p):
    return p if isinstance(p, str) else "custom:" + json.dumps(p["ranks"], sort_keys=False)


class CustomCF:
    """Per-day levers of a CF_SPEC file. `table` has one row per simulated day."""

    def __init__(self, spec_path, intervention_csv, num_steps, with_vacc=True):
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"CF_SPEC={spec_path!r} does not exist")
        self.spec_path = spec_path
        with open(spec_path) as f:
            spec = yaml.safe_load(f) or {}
        if not isinstance(spec, dict):
            raise ValueError(f"CF_SPEC {spec_path}: top level must be a mapping")
        bad = set(spec) - TOP_KEYS
        if bad:
            raise ValueError(f"CF_SPEC {spec_path}: unknown keys {sorted(bad)} (allowed {sorted(TOP_KEYS)})")
        vacc = spec.get("vaccine") or {}
        bad = set(vacc) - VACCINE_KEYS
        if bad:
            raise ValueError(f"CF_SPEC vaccine: unknown keys {sorted(bad)} (allowed {sorted(VACCINE_KEYS)})")
        self.name = spec_name(spec, spec_path)

        iv = pd.read_csv(intervention_csv)
        iv = iv[iv["t"] < num_steps].sort_values("t").reset_index(drop=True)
        if len(iv) != num_steps or list(iv["t"]) != list(range(num_steps)):
            raise RuntimeError(f"{intervention_csv}: needs one row for each t = 0..{num_steps - 1}")
        dates = [pd.Timestamp(d) for d in iv["date"]]

        school = _expand(spec.get("school"), dates, "factual", _lever, "school")
        work = _expand(spec.get("work"), dates, "factual", _work, "work")
        doses = vacc.get("doses", "factual")
        if doses == "factual":
            doses = None
        mult = _expand(doses, dates, 1.0, self._multiplier, "vaccine.doses")
        prio = _expand(vacc.get("schedule"), dates, "factual", _priority, "vaccine.schedule",
                       item_key="priority")

        has_65 = "vaccines_65plus" in iv.columns
        rows, self._days = [], []
        for j, r in iv.iterrows():
            f_school, f_occ = int(r["school_intervention"]), int(r["occ_intervention"])
            s = _resolve(school[j], f_school)
            w = work[j]
            if isinstance(w, dict):
                selective = True
                occ_e = _resolve(w["essential"], f_occ)
                occ_n = _resolve(w["nonessential"], f_occ)
                work_policy = f"essential={w['essential']};nonessential={w['nonessential']}"
            else:
                selective = False
                occ_e = occ_n = _resolve(w, f_occ)
                work_policy = w
            # WITH_VACC: false (e.g. the simple config) gives no doses whatever the spec says
            p = prio[j] if with_vacc else "none"
            m = 0.0 if p == "none" else mult[j]
            v = float(r["vaccines"]) * m
            v65 = float(r["vaccines_65plus"]) * m if has_65 else float("nan")
            self._days.append(dict(school=s, occ_essential=occ_e, occ_nonessential=occ_n,
                                   selective=selective, dose_multiplier=m, priority=p))
            rows.append({
                "date": r["date"], "school_intervention": s,
                # occ_intervention: the all-workplace value; under a split it is the essential value
                "occ_intervention": occ_e, "vaccines": v, "vaccines_65plus": v65, "t": int(r["t"]),
                "school_policy": school[j],
                "work_policy": work_policy,
                "work_closure_scope": "selective" if selective else "all",
                "occ_essential_intervention": occ_e,
                "occ_nonessential_intervention": occ_n,
                "vaccine_priority": _label(p) if with_vacc else "off (WITH_VACC: false)",
                "vaccine_dose_multiplier": m,
                "factual_school_intervention": f_school,
                "factual_occ_intervention": f_occ,
                "factual_vaccines": float(r["vaccines"]),
            })
        self.table = pd.DataFrame(rows)

    @staticmethod
    def _multiplier(v, where):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"CF_SPEC {where}: dose multiplier must be a number >= 0 "
                             f"(or 'factual' = 1), got {v!r}")
        return float(v)

    def day(self, t):
        return self._days[int(t)]

    def needs_essential(self):
        return any(d["selective"] or d["priority"] in ("essential_first", "hh_essential_home_first")
                   or (isinstance(d["priority"], dict) and set(d["priority"]["ranks"]) & set(WORK_KEYS))
                   for d in self._days)
