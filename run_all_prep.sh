#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=data_prep_all
#SBATCH --mail-user=facundoy@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=47G
#SBATCH --time=36:00:00
#SBATCH --partition=alrodri-a100
#SBATCH --output=/home/%u/%x-%j.log

# The application(s) to execute along with its input arguments and options:

echo "Starting data preparation for all counties..."
python3 scripts/run_all_prep.py
echo "Data preparation completed!"
