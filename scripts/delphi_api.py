import covidcast
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()

## BELOW IS THE EXAMPLE CODE - update your own api key
#################################################################
#################################################################

target_counties_env = os.getenv("TARGET_COUNTIES")
if target_counties_env:
    target_counties = [c.strip() for c in target_counties_env.split(",") if c.strip()]
else:
    target_counties = [
        '39013', '39081'
    ]

# query covidcast API for data
# Note:
#   smoothed_adj_cli is https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/doctor-visits.html
#   wcli (since April 15) from https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html
#   and many more if we use since 2020-09-08
covidcast.use_api_key(os.environ.get("COVIDCAST_API_KEY"))
start_date, end_date = date(2020, 6, 1), date(2021, 7, 31)

def process_daily_data():
    deaths = covidcast.signal(
        "indicator-combination",
        "deaths_incidence_num",
        start_date,
        end_date,
        "county",
        geo_values=target_counties
    )
    cases = covidcast.signal(
        "indicator-combination",
        "confirmed_incidence_num",
        start_date,
        end_date,
        "county",
        geo_values=target_counties
    )

    data = covidcast.aggregate_signals([cases, deaths])

    data = data.rename(
        columns={
            "indicator-combination_confirmed_incidence_num_0_value": "cases",
            "indicator-combination_deaths_incidence_num_1_value": "deaths",
        })

    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, "..")
    output_dir = os.path.join(project_root, "data", "delphi_county_data")
    output_dir = os.path.abspath(output_dir) 

    os.makedirs(output_dir, exist_ok=True)

    for fips_code in target_counties:
        county_data = data[data['geo_value'] == fips_code][[
            "time_value", "geo_value", "cases", "deaths"
        ]]
        
        output_file = os.path.join(output_dir, f"{fips_code}_data.csv")
        county_data.to_csv(output_file, index=False)

    print(data[[
        "time_value", "geo_value", "cases", "deaths"
    ]].head(20))

process_daily_data()
