#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=run_all_sims
#SBATCH --mail-user=facundoy@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=47G
#SBATCH --time=36:00:00
#SBATCH --partition=alrodri-a100
#SBATCH --array=0-153%3
#SBATCH --output=/home/%u/%x-%A-%a.log

# List of counties hardcoded in run_all_networks.py/run_all_prep.py
COUNTIES=(
    '01009' '01031' '01039' '01045' '01049' '01071' '13013' '13071' '13103' '13115'
    '13137' '13153' '17011' '17021' '17037' '17055' '17063' '17073' '17091' '17095'
    '17103' '17115' '17117' '17121' '17141' '17167' '19017' '19027' '19113' '19155'
    '19169' '20035' '20045' '20061' '20079' '20103' '20125' '20149' '20161' '20169'
    '21009' '21029' '21035' '21071' '21073' '21083' '21089' '21093' '21113' '21179'
    '21199' '21209' '21211' '22001' '22005' '22039' '22063' '26023' '26027' '26041'
    '26057' '26059' '26067' '26073' '26123' '27005' '27027' '27041' '27059' '27085'
    '27109' '29021' '29027' '29051' '29101' '30029' '30047' '30049' '30093' '36003'
    '36011' '36013' '36031' '36051' '38059' '38077' '38093' '39005' '39007' '39011'
    '39013' '39015' '39021' '39023' '39031' '39033' '39037' '39039' '39045' '39051'
    '39057' '39059' '39063' '39071' '39077' '39079' '39083' '39087' '40021' '40089'
    '40097' '40121' '40147' '42005' '42013' '42025' '42031' '42051' '42055' '42059'
    '42061' '42075' '42085' '42087' '46011' '46013' '46029' '46035' '46081' '46083'
    '46135' '47003' '47005' '47009' '47017' '48001' '48005' '48013' '48041' '48049'
    '48071' '48091' '48097' '48143' '48147' '48181' '48189' '48213' '51047' '51095'
    '51137' '51149' '51161'
)

# Verify array index is valid
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Error: This script must be submitted using 'sbatch' (requires SLURM_ARRAY_TASK_ID)."
    exit 1
fi

COUNTY=${COUNTIES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$COUNTY" ]; then
    echo "Error: Invalid county index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

PROJECT_ROOT=$(pwd)
WORK_DIR="workspace_task_${SLURM_ARRAY_TASK_ID}"

echo "Preparing workspace for county $COUNTY (Index $SLURM_ARRAY_TASK_ID)..."

# Ensure global folders exist
mkdir -p results result_graphs

# Create process-unique temporary workspace
mkdir -p "$WORK_DIR"
cd "$WORK_DIR" || exit 1

# Setup clean up of workspace directory on exit
cleanup() {
    echo "Cleaning up workspace..."
    cd "$PROJECT_ROOT" || exit
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Symlink all shared resources to prevent disk space duplication
ln -s ../populations populations
ln -s ../data data
ln -s ../results results
ln -s ../result_graphs result_graphs
ln -s ../agent_torch agent_torch
ln -s ../abm_nets.py abm_nets.py
ln -s ../main.py main.py
ln -s ../constants.py constants.py

# Copy the config folder (so this run has an isolated config.yaml)
cp -r ../covid_abm covid_abm

echo "Launching simulation for county: $COUNTY on GPU: $CUDA_VISIBLE_DEVICES"
python3 main.py "$COUNTY"

echo "Simulation completed for county: $COUNTY"
