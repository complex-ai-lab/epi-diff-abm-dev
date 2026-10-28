#!/bin/bash
# Calibrate -> run -> SEIRV check for the reference counties, split over the array tasks (run_counties.py).
#   sbatch run_counties.sbatch
#   sbatch --export=ALL,ARGS="--config config.yaml --run-name full" run_counties.sbatch
#SBATCH --job-name=abm_counties
#SBATCH --account=alrodri
#SBATCH --partition=alrodri-a100
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=47G
#SBATCH --time=12:00:00
#SBATCH --output=results/pipeline_logs/sbatch_%A_%a.log
export CPATH=/sw/pkgs/arc/python/3.11.5/include/python3.11   # Python.h for torch.compile (README section 1)
.venv/bin/python run_counties.py ${ARGS:-}
