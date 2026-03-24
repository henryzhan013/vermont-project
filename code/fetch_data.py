"""
Vermont Economic Hardship Analysis - Data Fetching Script
Fetches Census ACS 2022 data and Vermont school lunch data from Urban Institute API
"""

import requests
import pandas as pd
import json

# =============================================================================
# TASK 1: Census ACS 2022 County-Level Data for Vermont
# =============================================================================

print("=" * 60)
print("TASK 1: Fetching Census ACS 2022 Data for Vermont Counties")
print("=" * 60)

# Census API base URL
BASE_URL = "https://api.census.gov/data/2022/acs/acs5/subject"

# Variables to fetch:
# S1701_C03_001E - Percent below poverty level (all people)
# S1901_C01_012E - Median household income
# S2201_C03_001E - Percent of households receiving SNAP (this is actually a percentage)

variables = "NAME,S1701_C03_001E,S1901_C01_012E,S2201_C03_001E"

# Vermont state FIPS code is 50
params = {
    "get": variables,
    "for": "county:*",
    "in": "state:50"
}

print(f"\nFetching from: {BASE_URL}")
print(f"Variables: {variables}")

response = requests.get(BASE_URL, params=params)
print(f"Response status: {response.status_code}")

census_df = None

if response.status_code == 200:
    data = response.json()
    print(f"Rows returned: {len(data) - 1}")  # minus header row

    # Convert to DataFrame
    headers = data[0]
    rows = data[1:]

    census_df = pd.DataFrame(rows, columns=headers)

    # Rename columns for clarity
    census_df = census_df.rename(columns={
        'NAME': 'county_name',
        'S1701_C03_001E': 'poverty_rate_pct',
        'S1901_C01_012E': 'median_household_income',
        'S2201_C03_001E': 'snap_participation_pct',
        'state': 'state_fips',
        'county': 'county_fips'
    })

    # Clean county names (remove ", Vermont")
    census_df['county_name'] = census_df['county_name'].str.replace(', Vermont', '', regex=False)
    census_df['county_name'] = census_df['county_name'].str.replace(' County', '', regex=False)

    # Convert numeric columns
    numeric_cols = ['poverty_rate_pct', 'median_household_income', 'snap_participation_pct']
    for col in numeric_cols:
        census_df[col] = pd.to_numeric(census_df[col], errors='coerce')

    # Sort by county name
    census_df = census_df.sort_values('county_name').reset_index(drop=True)

else:
    print(f"Error fetching Census data: {response.status_code}")
    print(response.text)

# Now fetch SNAP percentage separately - the previous variable was actually count
# S2201_C04_001E is the percentage of households with SNAP benefits
print("\nFetching SNAP participation percentage...")
snap_url = "https://api.census.gov/data/2022/acs/acs5/subject"
snap_params = {
    "get": "NAME,S2201_C04_001E",
    "for": "county:*",
    "in": "state:50"
}

snap_response = requests.get(snap_url, params=snap_params)
if snap_response.status_code == 200:
    snap_data = snap_response.json()
    snap_headers = snap_data[0]
    snap_rows = snap_data[1:]
    snap_df = pd.DataFrame(snap_rows, columns=snap_headers)
    snap_df['county_name'] = snap_df['NAME'].str.replace(', Vermont', '', regex=False).str.replace(' County', '', regex=False)
    snap_df['snap_participation_pct'] = pd.to_numeric(snap_df['S2201_C04_001E'], errors='coerce')

    # Merge the correct SNAP percentage
    if census_df is not None:
        census_df = census_df.drop(columns=['snap_participation_pct'])
        census_df = census_df.merge(snap_df[['county_name', 'snap_participation_pct']], on='county_name', how='left')

if census_df is not None:
    print("\n" + "=" * 60)
    print("CENSUS DATA PREVIEW:")
    print("=" * 60)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_columns', 10)
    print(census_df.to_string(index=False))

    # Check for missing values
    numeric_cols = ['poverty_rate_pct', 'median_household_income', 'snap_participation_pct']
    print("\n" + "-" * 40)
    print("Missing Values Check:")
    print("-" * 40)
    missing = census_df[numeric_cols].isnull().sum()
    if missing.sum() == 0:
        print("No missing values in numeric columns.")
    else:
        print(missing)

    # Flag suspicious values
    print("\n" + "-" * 40)
    print("Data Quality Check:")
    print("-" * 40)
    issues = []
    for _, row in census_df.iterrows():
        county = row['county_name']
        if row['poverty_rate_pct'] > 25:
            issues.append(f"  - {county}: Poverty rate {row['poverty_rate_pct']}% seems high")
        if row['median_household_income'] < 40000:
            issues.append(f"  - {county}: Median income ${row['median_household_income']:,.0f} seems low")
        if row['snap_participation_pct'] > 25:
            issues.append(f"  - {county}: SNAP rate {row['snap_participation_pct']}% seems high")

    if issues:
        print("Potential issues flagged:")
        for issue in issues:
            print(issue)
    else:
        print("All values within expected ranges.")

    # Save to CSV
    census_df.to_csv('vermont_county_data.csv', index=False)
    print(f"\nSaved: vermont_county_data.csv ({len(census_df)} counties)")

# =============================================================================
# TASK 2: Vermont Free/Reduced Lunch Data from Urban Institute API
# =============================================================================

print("\n" + "=" * 60)
print("TASK 2: Fetching Vermont Free/Reduced Lunch Data")
print("=" * 60)

# Urban Institute Education Data API - CCD School Directory
# Contains free_lunch, reduced_price_lunch, and enrollment data
urban_url = "https://educationdata.urban.org/api/v1/schools/ccd/directory/2022/"
params = {
    "fips": 50,  # Vermont FIPS code
}

print(f"\nFetching from Urban Institute Education Data API...")
print(f"URL: {urban_url}")

# The API returns paginated results, so we need to fetch all pages
all_schools = []
next_url = f"{urban_url}?fips=50"

page = 1
while next_url:
    print(f"  Fetching page {page}...")
    response = requests.get(next_url)

    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        all_schools.extend(results)
        next_url = data.get('next')
        page += 1
    else:
        print(f"Error: {response.status_code}")
        break

print(f"Total schools fetched: {len(all_schools)}")

if all_schools:
    # Convert to DataFrame
    schools_df = pd.DataFrame(all_schools)

    # Select relevant columns
    columns_to_keep = [
        'school_name', 'lea_name', 'city_location',
        'enrollment', 'free_lunch', 'reduced_price_lunch',
        'free_or_reduced_price_lunch', 'school_level', 'school_status'
    ]

    # Check which columns exist
    available_cols = [c for c in columns_to_keep if c in schools_df.columns]
    schools_df = schools_df[available_cols].copy()

    # Filter to only open schools (school_status == 1 means open)
    if 'school_status' in schools_df.columns:
        schools_df = schools_df[schools_df['school_status'] == 1].copy()
        schools_df = schools_df.drop(columns=['school_status'])

    # Convert numeric columns
    numeric_cols = ['enrollment', 'free_lunch', 'reduced_price_lunch', 'free_or_reduced_price_lunch']
    for col in numeric_cols:
        if col in schools_df.columns:
            schools_df[col] = pd.to_numeric(schools_df[col], errors='coerce')

    # Calculate FRL percentage if we have the components
    if 'free_or_reduced_price_lunch' in schools_df.columns and 'enrollment' in schools_df.columns:
        schools_df['frl_pct'] = (schools_df['free_or_reduced_price_lunch'] / schools_df['enrollment'] * 100).round(1)

    # Clean up city names to serve as "town" approximation
    if 'city_location' in schools_df.columns:
        schools_df = schools_df.rename(columns={'city_location': 'town'})

    # Sort by town
    schools_df = schools_df.sort_values('town').reset_index(drop=True)

    print("\n" + "=" * 60)
    print("SCHOOL-LEVEL DATA PREVIEW (first 20 rows):")
    print("=" * 60)
    preview_cols = ['school_name', 'town', 'enrollment', 'free_or_reduced_price_lunch', 'frl_pct']
    preview_cols = [c for c in preview_cols if c in schools_df.columns]
    print(schools_df[preview_cols].head(20).to_string(index=False))

    # Aggregate to town level
    print("\n" + "-" * 40)
    print("Aggregating to Town Level...")
    print("-" * 40)

    # Group by town
    town_agg = schools_df.groupby('town').agg({
        'enrollment': 'sum',
        'free_or_reduced_price_lunch': 'sum',
        'school_name': 'count'
    }).reset_index()

    town_agg = town_agg.rename(columns={'school_name': 'num_schools'})
    town_agg['frl_pct'] = (town_agg['free_or_reduced_price_lunch'] / town_agg['enrollment'] * 100).round(1)
    town_agg = town_agg.rename(columns={
        'enrollment': 'total_enrollment',
        'free_or_reduced_price_lunch': 'frl_eligible'
    })

    # Sort by town
    town_agg = town_agg.sort_values('town').reset_index(drop=True)

    print("\n" + "=" * 60)
    print("TOWN-LEVEL DATA:")
    print("=" * 60)
    print(town_agg.to_string(index=False))

    # Check for missing/suspicious values
    print("\n" + "-" * 40)
    print("Missing Values Check:")
    print("-" * 40)
    missing = town_agg.isnull().sum()
    if missing.sum() == 0:
        print("No missing values.")
    else:
        print(missing)

    print("\n" + "-" * 40)
    print("Data Quality Check:")
    print("-" * 40)
    issues = []

    # Check for towns with suspiciously high or low FRL rates
    for _, row in town_agg.iterrows():
        town = row['town']
        if pd.isna(row['frl_pct']):
            issues.append(f"  - {town}: Missing FRL percentage (enrollment={row['total_enrollment']})")
        elif row['frl_pct'] > 90:
            issues.append(f"  - {town}: Very high FRL rate ({row['frl_pct']}%)")
        elif row['frl_pct'] == 0 and row['total_enrollment'] > 50:
            issues.append(f"  - {town}: Zero FRL with significant enrollment ({row['total_enrollment']})")
        if row['total_enrollment'] < 10:
            issues.append(f"  - {town}: Very small enrollment ({row['total_enrollment']} students)")

    if issues:
        print("Potential issues flagged:")
        for issue in issues[:15]:  # Limit output
            print(issue)
        if len(issues) > 15:
            print(f"  ... and {len(issues) - 15} more")
    else:
        print("All values within expected ranges.")

    # Save to CSV
    town_agg.to_csv('vermont_town_data.csv', index=False)
    print(f"\nSaved: vermont_town_data.csv ({len(town_agg)} towns)")

    # Also save school-level data for reference
    schools_df.to_csv('vermont_school_data.csv', index=False)
    print(f"Saved: vermont_school_data.csv ({len(schools_df)} schools)")

else:
    print("No school data retrieved.")

# =============================================================================
# TASK 3: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nFiles created:")
print("  1. vermont_county_data.csv - Census ACS 2022 county-level economic indicators")
print("  2. vermont_town_data.csv - Town-level FRL aggregated data")
print("  3. vermont_school_data.csv - School-level detail (for reference)")
print("\nData sources:")
print("  - Census ACS 2022 5-Year Estimates (api.census.gov)")
print("  - Urban Institute Education Data API - CCD 2022 (educationdata.urban.org)")
