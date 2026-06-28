# Commuter Networks

This folder contains utilities for creating commuter data networks in the COVID ABM simulation. The commuter networks represent agents who travel between counties for work, allowing for more realistic modeling of disease spread across metropolitan areas.

## Creating Commuter Networks

Initially, only the raw Excel file `residence_commute_data.xlsx` is provided in the repository. Before running the simulation, you must generate the formatted CSV file by executing the conversion script from the `networks/commuter_networks/` directory:

```bash
python3 commute_data/convert_xlsx_to_csv.py
```

This script will read the raw Excel file, clean the metadata and footnotes, format the FIPS codes, output `formatted_residence_commute_data.csv`, and automatically remove any temporary raw CSV files.

The main utility is `commute_data_utils.py`, which contains the `track_commute_data` function. This function:

1. Reads formatted residence commute data from `commute_data/formatted_residence_commute_data.csv`
2. For a given county, identifies agents who commute to other counties for work
3. Randomly selects agents with occupations to assign as commuters
4. Creates mapping files:
   - `{county}_residence_commute_data.txt`: Maps agent IDs to their workplace counties and is saved in the same directory as the occupation networks (`output_dir/mobility_networks/occnets/`).
5. Runs inline without returning a value (commuting agents are kept within the original population).

### Usage

```python
from commute_data_utils import track_commute_data

# individuals_df should contain columns 'ID' and 'Occupations'
track_commute_data(individuals_df, state_abbrev, county, output_dir)
```

This process ensures that agents commuting between counties are properly tracked and recorded in the mapping file without altering the local population dataframe.

## Data Source

You can access the relevant commuter data and similar datasets at:  
https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html
