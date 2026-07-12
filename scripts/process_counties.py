import pickle
import pandas as pd
import os
import json
import requests
from io import StringIO

state = os.getenv("TARGET_STATE", "OH")
target_counties_env = os.getenv("TARGET_COUNTIES")
if target_counties_env:
    target_counties = [c.strip() for c in target_counties_env.split(",") if c.strip()]
else:
    target_counties = [
        '39013', '39081'
    ]

# Load county policy database globally to avoid reloading it for each county
policy_db_path = os.path.join(os.path.dirname(__file__), "..", "data", "county_policy_data", "39109-0001-Data.tsv")
if os.path.exists(policy_db_path):
    print("Loading county policy database...")
    policy_db = pd.read_csv(policy_db_path, sep="\t")
    policy_db["WEEK_START_DATE"] = pd.to_datetime(policy_db["WEEK_START_DATE"], format="%d-%b-%Y")
    policy_db["WEEK_END_DATE"] = pd.to_datetime(policy_db["WEEK_END_DATE"], format="%d-%b-%Y")
else:
    print(f"Warning: County policy database not found at {policy_db_path}")
    policy_db = None

def process_fips_data(fips_code):
    # Define paths
    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_daily_path = os.path.join(project_root, "data", "delphi_county_data", f"{fips_code}_data.csv")
    output_dir = os.path.join(project_root, "data", "processed_data", fips_code, "202010-202104")
    os.makedirs(output_dir, exist_ok=True)

    # Define final date range to save
    final_start = pd.to_datetime("2020-10-26")
    final_end   = pd.to_datetime("2021-04-11")

    # Helper: replace a negative value with the average of nearest non-negative neighbors
    def fix_negative(series):
        series = series.copy()
        for i in range(len(series)):
            if series[i] < 0:
                # find previous non-negative
                prev_idx = i - 1
                while prev_idx >= 0 and series[prev_idx] < 0:
                    prev_idx -= 1
                # find next non-negative
                next_idx = i + 1
                while next_idx < len(series) and series[next_idx] < 0:
                    next_idx += 1
                # assign replacement
                if prev_idx >= 0 and next_idx < len(series):
                    series[i] = (series[prev_idx] + series[next_idx]) / 2
                elif prev_idx >= 0:
                    series[i] = series[prev_idx]
                elif next_idx < len(series):
                    series[i] = series[next_idx]
                else:
                    series[i] = 0
        return series

    # Process daily
    if os.path.exists(input_daily_path):
        # Load all of 2020 and 2021
        df = pd.read_csv(input_daily_path, parse_dates=["time_value"])
        df = df[df["time_value"].dt.year.isin([2020, 2021])]
        df = df[["time_value", "cases", "deaths"]].rename(columns={"time_value": "date"})
        df = df.sort_values("date").reset_index(drop=True)

        # Fix negative values
        df["cases"]  = fix_negative(df["cases"])
        df["deaths"] = fix_negative(df["deaths"])

        # Compute 7-day sliding window average for cases (3 days before, actual day, 3 days after)
        df["cases"] = df["cases"].rolling(window=7, center=True, min_periods=1).mean()

        # Aggregate weekly on full data
        df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly_df = df.groupby("week", as_index=False)["deaths"].sum()
        weekly_df = weekly_df.rename(columns={"week": "date"})

        # Slice to the final desired range before saving
        df_final    = df[(df["date"] >= final_start) & (df["date"] <= final_end)]
        weekly_final = weekly_df[(weekly_df["date"] >= final_start) & (weekly_df["date"] <= final_end)]

        # Save daily
        daily_output_path = os.path.join(output_dir, "daily_data.csv")
        df_final.to_csv(daily_output_path, index=False)
        print(f"Saved daily data to {daily_output_path}")

        # Save weekly
        weekly_output_path = os.path.join(output_dir, "weekly_data.csv")
        weekly_final.to_csv(weekly_output_path, index=False)
        print(f"Saved weekly data to {weekly_output_path}")

    else:
        print(f"Daily data file not found for FIPS {fips_code}.")


def create_county_folder(state_abbrev, fips_code):
    # Construct the county directory name using state abbreviation + FIPS
    county_dir_name = f"pop{fips_code}"
    county_dir = os.path.join('./populations', county_dir_name)
    os.makedirs(county_dir, exist_ok=True)

    # Create mapping.json
    mapping_data = {
        "age": [
            "20t29",
            "30t39",
            "40t49",
            "50t64",
            "65A",
            "U19"
        ]
    }
    with open(os.path.join(county_dir, 'mapping.json'), 'w') as f:
        json.dump(mapping_data, f, indent=2)


    mobility_dir = os.path.join(county_dir, "mobility_networks")
    os.makedirs(mobility_dir, exist_ok=True)
    zero_file = os.path.join(mobility_dir, "0.csv")
    pd.DataFrame([[0, 0]]).to_csv(zero_file, index=False, header=False)

    # Build population file path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    pop_file = f"population_data/{state_abbrev}_population_data/{fips_code}_population.csv"
    pop_file = os.path.join(data_dir, pop_file)
    
    try:
        df = pd.read_csv(pop_file)

        # Create age.csv
        df['Age'].to_csv(os.path.join(county_dir, 'age.csv'), index=False, header=['age'])

        # Create age.pickle
        age_mapping = []
        for age in df['Age']:
            if age < 20:
                mapped = 5  # U19
            elif 20 <= age < 30:
                mapped = 0  # 20t29
            elif 30 <= age < 40:
                mapped = 1  # 30t39
            elif 40 <= age < 50:
                mapped = 2  # 40t49
            elif 50 <= age < 65:
                mapped = 3  # 50t64
            else:
                mapped = 4  # 65A
            age_mapping.append(mapped)

        # Create pandas Series with name 'age'
        age_series = pd.Series(age_mapping, name='age')
        with open(os.path.join(county_dir, 'age.pickle'), 'wb') as f:
            pickle.dump(age_series, f)

        # Create disease_stages.csv
        stages = [0.0] * len(df)
        pd.DataFrame(stages, columns=['stages']).to_csv(
            os.path.join(county_dir, 'disease_stages.csv'),
            index=False
        )

        print(f"Successfully created files for {county_dir_name}")

    except FileNotFoundError:
        print(f"Population file not found: {pop_file}")
    except Exception as e:
        print(f"Error processing {county_dir_name}: {str(e)}")

    # Generate the intervention file

    start_date = pd.to_datetime("2020-10-26")
    dates = pd.date_range(start_date, periods=182, freq='D')
    interventions_df = pd.DataFrame({"date": dates})

    fips_int = int(fips_code)
    county_policy = None
    county_policy_found = False

    if policy_db is not None:
        # 1. Try to find county-level data in the policy database
        county_policy = policy_db[policy_db["COUNTY_FIPS"] == fips_int].copy()
        if len(county_policy) > 0:
            county_policy = county_policy.sort_values("WEEK_START_DATE").reset_index(drop=True)
            county_policy_found = True
            print(f"Using county-level policy data for FIPS {fips_code}")

    if not county_policy_found:
        print(f"FIPS {fips_code} not found in county policy database. Falling back to state-level policies for {state_abbrev}.")
        # 2. Fallback: Filter by state and group by week to get state-level mandates
        if policy_db is not None:
            state_policy = policy_db[policy_db["STATE"] == state_abbrev].copy()
            state_policy = state_policy.sort_values("WEEK_START_DATE").reset_index(drop=True)
            county_policy = state_policy.groupby("WEEK_START_DATE", as_index=False).first()

    def map_intervention_val(val):
        if pd.isna(val) or val in [99, 77]:
            return 0
        return 0 if val in [0, 1] else 1

    school_vals = []
    occ_vals = []

    for d in interventions_df["date"]:
        if county_policy is not None and len(county_policy) > 0:
            # Find the week matching this day
            week_row = county_policy[(county_policy["WEEK_START_DATE"] <= d) & (county_policy["WEEK_END_DATE"] >= d)]
            if len(week_row) > 0:
                row = week_row.iloc[0]
                
                if county_policy_found:
                    # Use county-level columns, fallback to state-level only if missing (99)
                    school_val = row["C1_SCHOOL"]
                    if pd.isna(school_val) or school_val == 99:
                        school_val = row["S_C1_SCHOOL"]
                    
                    occ_val = row["C2_WORKPLACE"]
                    if pd.isna(occ_val) or occ_val == 99:
                        occ_val = row["S_C2_WORKPLACE"]
                else:
                    # State-level fallback: directly use state columns
                    school_val = row["S_C1_SCHOOL"]
                    occ_val = row["S_C2_WORKPLACE"]
                
                school_vals.append(map_intervention_val(school_val))
                occ_vals.append(map_intervention_val(occ_val))
            else:
                school_vals.append(0)
                occ_vals.append(0)
        else:
            school_vals.append(0)
            occ_vals.append(0)

    interventions_df["school_intervention"] = school_vals
    interventions_df["occ_intervention"] = occ_vals

    # Vaccination Data
    fips_str = str(fips_code).zfill(5)
    url = (
        "https://data.cdc.gov/resource/8xkx-amqh.csv?"
        f"$select=date,fips,administered_dose1_recip"
        f"&fips={fips_str}"
        f"&$limit=50000"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        interventions_df["vaccines"] = 0
    else:
        cdc_df = pd.read_csv(StringIO(response.text))
        cdc_df["date"] = pd.to_datetime(cdc_df["date"])
        
        cdc_df["fips"] = cdc_df["fips"].astype(str).str.zfill(5)
        cdc_df = cdc_df[cdc_df["fips"] == fips_str].copy()

        cdc_df.sort_values("date", inplace=True)
        cdc_df["new_vax"] = cdc_df["administered_dose1_recip"].diff().fillna(0)

        merged = interventions_df.merge(
            cdc_df[["date", "new_vax"]],
            on="date",
            how="left"
        )
        interventions_df["vaccines"] = merged["new_vax"].fillna(0).values

    # convert date to string for output
    interventions_df["date"] = interventions_df["date"].dt.strftime("%Y-%m-%d")

    interventions_df["t"] = range(len(interventions_df))

    # save intervention.csv inside county_dir
    interventions_df.to_csv(
        os.path.join(county_dir, "intervention.csv"),
        index=False
    )

    print(f"Created intervention.csv for {county_dir_name}")


def generate_all_county_data(state_abbrv, county):
    process_fips_data(county)
    create_county_folder(state_abbrv, county)


for county in target_counties:
    generate_all_county_data(state, county)
