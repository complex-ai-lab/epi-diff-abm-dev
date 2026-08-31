"""
Presentation plot: counterfactual daily-case trajectories for the fixed_60
(vaccinated-state, R->S at 60d) runs, one figure per county.

Each figure overlays CF types 1, 2, 3, 4, 7, 11 (each a distinct colour) on top
of the observed ("actual") daily cases drawn as a translucent grey line.

Data source (already generated):
  vaccinated_test/results/<FIPS>/counterfactuals/fixed_60/all_counterfactual_results/
      <NN_folder>/data/<FIPS>_counterfactual_data<TYPE>.csv
  columns: day, counterfactual_cases, actual_cases
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

COUNTIES = {
    "17117": "Tazewell County, IL",
    "36003": "Allegany County, NY",
    "01031": "Coffee County, AL",
}

# CF type -> (folder name, human-readable label, colour)
CF_TYPES = {
    1:  ("01_static_all_open",                 "Type 1: All open (school open, work open)",   "#d62728"),
    2:  ("02_static_school_closed_work_open",   "Type 2: School closed, work open",            "#ff7f0e"),
    3:  ("03_static_school_open_work_closed",   "Type 3: School open, work closed",            "#1f77b4"),
    4:  ("04_static_all_closed",                "Type 4: All closed (school + work closed)",   "#2ca02c"),
    7:  ("07_timevar_factual_school_factual_work", "Type 7: Factual baseline (matches history)", "#000000"),
    11: ("11_timevar_factual_no_vaccines",      "Type 11: Factual baseline, NO vaccines",      "#9467bd"),
}

BASE = "vaccinated_test/results/{fips}/counterfactuals/fixed_60/all_counterfactual_results"
OUT_DIR = "result_graphs/pres_plots"


def load(fips, folder, cf_type):
    path = os.path.join(
        BASE.format(fips=fips), folder, "data",
        f"{fips}_counterfactual_data{cf_type}.csv",
    )
    return pd.read_csv(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for fips, name in COUNTIES.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # actual cases (same series in every CF csv) -> translucent grey
        actual = load(fips, CF_TYPES[7][0], 7)
        ax.plot(actual["day"], actual["actual_cases"],
                color="gray", alpha=0.5, lw=2.0, label="Actual cases", zorder=1)

        for cf_type, (folder, label, colour) in CF_TYPES.items():
            df = load(fips, folder, cf_type)
            ax.plot(df["day"], df["counterfactual_cases"],
                    color=colour, lw=1.6, label=label, zorder=2)

        ax.set_title(f"Counterfactual daily cases  |  {fips} ({name})  |  fixed_60 vaccinated run")
        ax.set_xlabel("Simulation day")
        ax.set_ylabel("Daily cases")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"pres_cf_{fips}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
