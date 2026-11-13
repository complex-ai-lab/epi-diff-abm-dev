"""Calibrate -> run -> SEIRV check for several counties (section 10 of the README, one county after another).

    .venv/bin/python run_counties.py --config config.yaml --run-name simple
    .venv/bin/python run_counties.py --config config.yaml --run-name simple --counties 13013 17037
    sbatch run_counties.sbatch                      # 3 GPUs; the counties are split over the array tasks

Which counties: --counties if given, else run_settings.COUNTIES in the config:
  all                  every county with populations/pop<FIPS>, data/processed_data/<FIPS>/<DATE>/daily_data.csv
                       and networks (data/networks/covid_output_p0/<FIPS> or $NETWORKS_DIR_FALLBACK/<FIPS>)
  "01031 13013"        these FIPS codes (quote them in YAML, or leading zeros are lost)
--counties takes the same two forms (--counties all, --counties 01031 13013).

Per county:
  1. calibrate.sh   skipped (SKIP_EXISTING=1) if the calibration's calibrated_params.txt exists
  2. run.sh         -> results/<FIPS>/<RUN_NAME>/ and reference_results_<RUN_NAME>/<FIPS>/;
                    skipped if reference_results_<RUN_NAME>/<FIPS>/seir_rates_daily.csv exists
  3. seirv_check.py -> results/seirv_check/reference_results_<RUN_NAME>/<FIPS>/
Counties need populations/pop<FIPS> and data/processed_data/<FIPS>; their networks are read from
data/networks/covid_output_p0/<FIPS>, else from $NETWORKS_DIR_FALLBACK (the scratch copy of the P0 networks).
A failed county is reported and skipped; the script exits 1 at the end if any failed.
Logs: results/pipeline_logs/<RUN_NAME>_<step>_<FIPS>.log. With SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
set, this task does counties[task::count]; with --queue DIR the tasks instead claim counties one at a time
from the shared list (claim files in DIR), so a task that starts late or runs fast takes more.
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
P0_NETWORKS = "/scratch/alrodri_root/alrodri/pkbiswas/epi-diff-abm-dev/networks/covid_output_p0"


def step(name, cmd, env, log):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"{name} exited {rc}, see {log}")
    return time.time() - t0


def run_county(fips, args, py):
    env = dict(os.environ, FIPS=fips, CONFIG=args.config, RUN_NAME=args.run_name, PY=py)
    for need in (f"populations/pop{fips}", f"data/processed_data/{fips}"):
        if not os.path.exists(os.path.join(ROOT, need)):
            raise RuntimeError(f"missing {need}")
    if not os.path.isdir(os.path.join(ROOT, "data/networks/covid_output_p0", fips)):
        nets = os.environ.get("NETWORKS_DIR_FALLBACK", P0_NETWORKS)
        if not os.path.isdir(os.path.join(nets, fips)):
            raise RuntimeError(f"no networks for {fips} in data/networks/covid_output_p0 or {nets}")
        env["NETWORKS_DIR"] = nets
    logs = os.path.join(ROOT, "results", "pipeline_logs")
    os.makedirs(logs, exist_ok=True)
    log = lambda s: os.path.join(logs, f"{args.run_name}_{s}_{fips}.log")
    ref = f"reference_results_{args.run_name}/{fips}"
    done = []

    secs = step("calibrate", ["bash", "calibrate.sh"], dict(env, SKIP_EXISTING="1"), log("calibrate"))
    done.append(f"calibrate {secs / 60:.0f} min")
    if os.path.exists(os.path.join(ROOT, ref, "seir_rates_daily.csv")):
        done.append("run skipped")
    else:
        secs = step("run", ["bash", "run.sh"], dict(env, OVERWRITE="1"), log("run"))
        done.append(f"run {secs / 60:.0f} min")
    step("seirv_check", [py, "seirv_check.py", "--rates", f"{ref}/seir_rates_daily.csv"], env, log("seirv_check"))
    with open(log("seirv_check")) as f:
        return ", ".join(done) + " | " + f.read().strip().splitlines()[-1]


def has_inputs(fips, date):
    nets = os.environ.get("NETWORKS_DIR_FALLBACK", P0_NETWORKS)
    return (os.path.isdir(os.path.join(ROOT, "populations", f"pop{fips}"))
            and os.path.isfile(os.path.join(ROOT, "data/processed_data", fips, date, "daily_data.csv"))
            and any(os.path.isdir(os.path.join(d, fips, "mobility_networks"))
                    for d in (os.path.join(ROOT, "data/networks/covid_output_p0"), nets)))


def select_counties(cli, config):
    """--counties, else run_settings.COUNTIES of the config: 'all' or FIPS codes."""
    with open(os.path.join(ROOT, config)) as f:
        cfg = yaml.safe_load(f)
    if cli:
        spec = cli
    else:
        if "COUNTIES" not in (cfg.get("run_settings") or {}):
            sys.exit(f"[counties] {config} has no run_settings.COUNTIES; set it to all or FIPS codes, or pass --counties")
        spec = str(cfg["run_settings"]["COUNTIES"]).replace(",", " ").split()
    if spec == ["all"]:
        date = cfg["simulation_metadata"]["DATE"]
        pops = sorted(d[3:] for d in os.listdir(os.path.join(ROOT, "populations")) if d.startswith("pop") and d[3:].isdigit())
        excluded = str((cfg.get("run_settings") or {}).get("EXCLUDE_COUNTIES") or "").replace(",", " ").split()
        if excluded:
            print(f"[counties] all: excluding {len(excluded)} from EXCLUDE_COUNTIES: {' '.join(excluded)}", flush=True)
        pops = [f for f in pops if f not in excluded]
        missing = [f for f in pops if not has_inputs(f, date)]
        if missing:
            print(f"[counties] all: skipping {len(missing)} without processed data or networks: {' '.join(missing)}",
                  flush=True)
        return [f for f in pops if f not in missing]
    bad = [f for f in spec if not (len(f) == 5 and f.isdigit())]
    if bad:
        sys.exit(f"[counties] not 5-digit FIPS codes: {bad} (quote COUNTIES in the YAML so leading zeros survive)")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--counties", nargs="*", help="FIPS codes or 'all' (default: run_settings.COUNTIES of the config)")
    ap.add_argument("--queue", metavar="DIR",
                    help="shared claim folder: every array task walks the whole list and runs the counties it claims "
                         "first (balances tasks that start at different times); use a new folder per submission")
    args = ap.parse_args()
    py = os.environ.get("PY", os.path.join(ROOT, ".venv/bin/python"))
    counties = select_counties(args.counties, args.config)
    task, count = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if args.queue:
        os.makedirs(args.queue, exist_ok=True)
        print(f"[counties] task {task}/{count}: shared queue {args.queue} over {' '.join(counties)}", flush=True)
    else:
        counties = counties[task::count]
        print(f"[counties] task {task}/{count}: {' '.join(counties)}", flush=True)

    failed = []
    for fips in counties:
        if args.queue:
            try:  # atomic claim: the first task to create the file runs this county
                os.close(os.open(os.path.join(args.queue, fips), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        try:
            print(f"[counties] {fips} OK: {run_county(fips, args, py)}", flush=True)
        except Exception as e:
            failed.append(fips)
            print(f"[counties] {fips} FAILED: {e}", flush=True)
    if failed:
        print(f"[counties] failed: {' '.join(failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
