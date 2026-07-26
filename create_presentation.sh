#!/bin/bash
# create_presentation.sh
# Shell script to generate a PowerPoint presentation summarizing county simulation results.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Starting PowerPoint Presentation Generation"
echo "============================================================"

# Ensure required Python packages are installed in the active Conda environment
python3 -c "import pptx, yaml, pandas, numpy" 2>/dev/null || {
    echo "Installing missing packages in Conda environment..."
    if command -v conda &> /dev/null && [ -n "$CONDA_PREFIX" ]; then
        echo "Active Conda environment detected ($CONDA_DEFAULT_ENV: $CONDA_PREFIX)"
        conda install -y -c conda-forge python-pptx pyyaml pandas numpy 2>/dev/null || python3 -m pip install python-pptx pyyaml pandas numpy
    else
        python3 -m pip install python-pptx pyyaml pandas numpy
    fi
}

OUTPUT_FILE="${1:-simulation_results_presentation.pptx}"

echo "Running presentation generator script..."
python3 build_presentation.py "$OUTPUT_FILE"

echo "============================================================"
echo " Done! Presentation saved as: $OUTPUT_FILE"
echo "============================================================"
