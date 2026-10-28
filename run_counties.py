"""Calibrate -> run -> SEIRV check for several counties (section 10 of the README, one county after another).

    .venv/bin/python run_counties.py --config config_simple.yaml --run-name simple
    .venv/bin/python run_counties.py --config config_simple.yaml --run-name simple --counties 13013 17037
    sbatch run_counties.sbatch                      # 3 GPUs; the counties are split over the array tasks

Per county (default: every folder in reference_results/):
  1. calibrate.sh   skipped (SKIP_EXISTING=1) if the calibration's calibrated_params.txt exists
  2. run.sh         -> results/<FIPS>/<RUN_NAME>/ and reference_results_<RUN_NAME>/<FIPS>/;
                    skipped if reference_results_<RUN_NAME>/<FIPS>/seir_rates_daily.csv exists
  3. seirv_check.py -> results/seirv_check/reference_results_<RUN_NAME>/<FIPS>/
Counties need populations/pop<FIPS> and data/processed_data/<FIPS>; their networks are read from
data/networks/covid_output_p0/<FIPS>, else from $NETWORKS_DIR_FALLBACK (the scratch copy of the P0 networks).
A failed county is reported and skipped; the script exits 1 at the end if any failed.
Logs: results/pipeline_logs/<RUN_NAME>_<step>_<FIPS>.log. With SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
set, this task does counties[task::count].
"""
import argparse
import os
import subprocess
import sys
import time

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_simple.yaml")
    ap.add_argument("--run-name", default="simple")
    ap.add_argument("--counties", nargs="*", help="FIPS codes (default: the folders in reference_results/)")
    args = ap.parse_args()
    py = os.environ.get("PY", os.path.join(ROOT, ".venv/bin/python"))
    counties = args.counties or sorted(os.listdir(os.path.join(ROOT, "reference_results")))
    task, count = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    counties = counties[task::count]
    print(f"[counties] task {task}/{count}: {' '.join(counties)}", flush=True)

    failed = []
    for fips in counties:
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
