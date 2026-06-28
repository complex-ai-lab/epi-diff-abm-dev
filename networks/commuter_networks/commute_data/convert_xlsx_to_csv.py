import os
import pandas as pd

def convert_xlsx_to_csv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(current_dir, "residence_commute_data.xlsx")
    formatted_csv_path = os.path.join(current_dir, "formatted_residence_commute_data.csv")
    raw_csv_path = os.path.join(current_dir, "residence_commute_data.csv")

    print(f"Reading Excel file from: {xlsx_path}")
    # The actual column headers are located on the 8th row (index 7 in 0-indexed count)
    df = pd.read_excel(xlsx_path, header=7)

    print("Initial row count:", len(df))

    # The footer rows contain notes/source info. Drop rows where 'State FIPS Code' is missing/non-numeric.
    df = df.dropna(subset=['State FIPS Code'])
    df['State FIPS Code'] = df['State FIPS Code'].astype(str).str.strip()
    df = df[df['State FIPS Code'].str.isdigit()]

    # Drop rows where workplace FIPS codes are missing/NaN (e.g. workplaces outside the U.S.)
    df = df.dropna(subset=['State FIPS Code.1', 'County FIPS Code.1'])
    print("Row count after dropping missing workplace FIPS:", len(df))

    # Helper function to format state and county FIPS codes to 5-digit codes
    def make_fips(state_col, county_col):
        state_str = state_col.astype(float).astype(int).astype(str).str.zfill(2)
        county_str = county_col.astype(float).astype(int).astype(str).str.zfill(3)
        return state_str + county_str

    print("Constructing 5-digit FIPS codes...")
    residence_county = make_fips(df['State FIPS Code'], df['County FIPS Code'])
    work_county = make_fips(df['State FIPS Code.1'], df['County FIPS Code.1'])
    num_commuters = df['Workers in Commuting Flow'].astype(float).astype(int)

    formatted_df = pd.DataFrame({
        'residence_county': residence_county,
        'work_county': work_county,
        'num_commuters': num_commuters
    })

    print(f"Saving formatted CSV to: {formatted_csv_path}")
    formatted_df.to_csv(formatted_csv_path, index=False)
    print("Conversion and formatting completed successfully!")

    # Clean up the unformatted raw CSV if it exists
    if os.path.exists(raw_csv_path):
        print(f"Removing unformatted raw CSV at: {raw_csv_path}")
        os.remove(raw_csv_path)
        print("Raw CSV cleanup completed.")

if __name__ == "__main__":
    convert_xlsx_to_csv()
