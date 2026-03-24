"""
Step 2: Build Comprehensive Town-Level Feature Matrix
Aggregates school data, merges with county data, creates derived features
"""

import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

county_df = pd.read_csv('vermont_county_data.csv')
town_df = pd.read_csv('vermont_town_data.csv')
school_df = pd.read_csv('vermont_school_data.csv')

print(f"Counties: {len(county_df)}")
print(f"Towns: {len(town_df)}")
print(f"Schools: {len(school_df)}")

# =============================================================================
# VERMONT TOWN-TO-COUNTY MAPPING
# =============================================================================
# Vermont has 251 incorporated towns across 14 counties
# This mapping covers towns that appear in our school data

TOWN_TO_COUNTY = {
    # Addison County
    'Addison': 'Addison', 'Bridport': 'Addison', 'Bristol': 'Addison',
    'Cornwall': 'Addison', 'Ferrisburgh': 'Addison', 'Goshen': 'Addison',
    'Granville': 'Addison', 'Hancock': 'Addison', 'Leicester': 'Addison',
    'Lincoln': 'Addison', 'Middlebury': 'Addison', 'Monkton': 'Addison',
    'New Haven': 'Addison', 'Orwell': 'Addison', 'Panton': 'Addison',
    'Ripton': 'Addison', 'Salisbury': 'Addison', 'Shoreham': 'Addison',
    'Starksboro': 'Addison', 'Vergennes': 'Addison', 'Waltham': 'Addison',
    'Weybridge': 'Addison', 'Whiting': 'Addison',

    # Bennington County
    'Arlington': 'Bennington', 'Bennington': 'Bennington', 'Dorset': 'Bennington',
    'Glastenbury': 'Bennington', 'Landgrove': 'Bennington', 'Manchester': 'Bennington',
    'Manchester Center': 'Bennington', 'Peru': 'Bennington', 'Pownal': 'Bennington',
    'Readsboro': 'Bennington', 'Rupert': 'Bennington', 'Sandgate': 'Bennington',
    'Searsburg': 'Bennington', 'Shaftsbury': 'Bennington', 'Stamford': 'Bennington',
    'Stramford': 'Bennington',  # alternate spelling in data
    'Sunderland': 'Bennington', 'Winhall': 'Bennington', 'Woodford': 'Bennington',

    # Caledonia County
    'Barnet': 'Caledonia', 'Burke': 'Caledonia', 'Danville': 'Caledonia',
    'Groton': 'Caledonia', 'Hardwick': 'Caledonia', 'Kirby': 'Caledonia',
    'Lyndon': 'Caledonia', 'Lyndonville': 'Caledonia', 'Newark': 'Caledonia',
    'Peacham': 'Caledonia', 'Ryegate': 'Caledonia', 'Sheffield': 'Caledonia',
    'St. Johnsbury': 'Caledonia', 'Saint Johnsbury': 'Caledonia',
    'Stannard': 'Caledonia', 'Sutton': 'Caledonia', 'Walden': 'Caledonia',
    'Waterford': 'Caledonia', 'West Burke': 'Caledonia', 'Wheelock': 'Caledonia',

    # Chittenden County
    'Bolton': 'Chittenden', 'Burlington': 'Chittenden', 'Charlotte': 'Chittenden',
    'Colchester': 'Chittenden', 'Essex': 'Chittenden', 'Essex Junction': 'Chittenden',
    'Hinesburg': 'Chittenden', 'Huntington': 'Chittenden', 'Jericho': 'Chittenden',
    'Jeffersonville': 'Chittenden',  # Cambridge is in Lamoille but Jeffersonville listed separately
    'Milton': 'Chittenden', 'Richmond': 'Chittenden', 'Shelburne': 'Chittenden',
    'South Burlington': 'Chittenden', 'St. George': 'Chittenden',
    'Underhill': 'Chittenden', 'Underhill Center': 'Chittenden',
    'Westford': 'Chittenden', 'Williston': 'Chittenden', 'Winooski': 'Chittenden',

    # Essex County
    'Averill': 'Essex', 'Bloomfield': 'Essex', 'Brighton': 'Essex',
    'Brunswick': 'Essex', 'Canaan': 'Essex', 'Concord': 'Essex',
    'East Haven': 'Essex', 'Ferdinand': 'Essex', 'Gilman': 'Essex',
    'Granby': 'Essex', 'Guildhall': 'Essex', 'Island Pond': 'Essex',
    'Lemington': 'Essex', 'Lewis': 'Essex', 'Lunenburg': 'Essex',
    'Maidstone': 'Essex', 'Norton': 'Essex', 'Victory': 'Essex',

    # Franklin County
    'Bakersfield': 'Franklin', 'Berkshire': 'Franklin', 'Enosburg': 'Franklin',
    'Enosburg Falls': 'Franklin', 'Fairfax': 'Franklin', 'Fairfield': 'Franklin',
    'Fletcher': 'Franklin', 'Franklin': 'Franklin', 'Georgia': 'Franklin',
    'Highgate': 'Franklin', 'Highgate Center': 'Franklin',
    'Montgomery': 'Franklin', 'Montgomery Center': 'Franklin',
    'Richford': 'Franklin', 'St. Albans': 'Franklin', 'Saint Albans': 'Franklin',
    'Sheldon': 'Franklin', 'Swanton': 'Franklin',

    # Grand Isle County
    'Alburg': 'Grand Isle', 'Alburgh': 'Grand Isle',
    'Grand Isle': 'Grand Isle', 'Isle La Motte': 'Grand Isle',
    'North Hero': 'Grand Isle', 'South Hero': 'Grand Isle',

    # Lamoille County
    'Belvidere': 'Lamoille', 'Cambridge': 'Lamoille', 'Eden': 'Lamoille',
    'Elmore': 'Lamoille', 'Hyde Park': 'Lamoille', 'Johnson': 'Lamoille',
    'Lake Elmore': 'Lamoille',  # alternate name
    'Morristown': 'Lamoille', 'Morrisville': 'Lamoille',
    'Stowe': 'Lamoille', 'Waterville': 'Lamoille', 'Wolcott': 'Lamoille',

    # Orange County
    'Bradford': 'Orange', 'Braintree': 'Orange', 'Brookfield': 'Orange',
    'Chelsea': 'Orange', 'Corinth': 'Orange', 'East Corinth': 'Orange',
    'Fairlee': 'Orange', 'Newbury': 'Orange', 'Orange': 'Orange',
    'Randolph': 'Orange', 'Strafford': 'Orange', 'South Strafford': 'Orange',
    'Thetford': 'Orange', 'Topsham': 'Orange', 'Tunbridge': 'Orange',
    'Vershire': 'Orange', 'Washington': 'Orange', 'West Fairlee': 'Orange',
    'Williamstown': 'Orange', 'Wells River': 'Orange',

    # Orleans County
    'Albany': 'Orleans', 'Barton': 'Orleans', 'Brownington': 'Orleans',
    'Charleston': 'Orleans', 'West Charleston': 'Orleans',
    'Coventry': 'Orleans', 'Craftsbury': 'Orleans', 'Craftsbury Common': 'Orleans',
    'Derby': 'Orleans', 'Derby Line': 'Orleans',
    'Glover': 'Orleans', 'Greensboro': 'Orleans', 'Holland': 'Orleans',
    'Irasburg': 'Orleans', 'Jay': 'Orleans', 'Lowell': 'Orleans',
    'Morgan': 'Orleans', 'Newport': 'Orleans', 'Newport Center': 'Orleans',
    'North Troy': 'Orleans', 'Troy': 'Orleans',
    'Orleans': 'Orleans', 'Westfield': 'Orleans', 'Westmore': 'Orleans',

    # Rutland County
    'Benson': 'Rutland', 'Brandon': 'Rutland', 'Castleton': 'Rutland',
    'Chittenden': 'Rutland', 'Clarendon': 'Rutland', 'North Clarendon': 'Rutland',
    'Cuttingsville': 'Rutland',  # part of Shrewsbury
    'Danby': 'Rutland', 'Fair Haven': 'Rutland', 'Hubbardton': 'Rutland',
    'Ira': 'Rutland', 'Killington': 'Rutland', 'Mendon': 'Rutland',
    'Middletown Springs': 'Rutland', 'Mount Holly': 'Rutland',
    'Mount Tabor': 'Rutland', 'Pawlet': 'Rutland', 'West Pawlet': 'Rutland',
    'Pittsfield': 'Rutland', 'Pittsford': 'Rutland', 'Poultney': 'Rutland',
    'Proctor': 'Rutland', 'Rutland': 'Rutland', 'Rutland Town': 'Rutland',
    'Shrewsbury': 'Rutland', 'Sudbury': 'Rutland', 'Tinmouth': 'Rutland',
    'Wallingford': 'Rutland', 'Wells': 'Rutland', 'West Haven': 'Rutland',
    'West Rutland': 'Rutland',

    # Washington County
    'Barre': 'Washington', 'Barre City': 'Washington', 'Barre Town': 'Washington',
    'Berlin': 'Washington', 'Cabot': 'Washington', 'Calais': 'Washington',
    'Duxbury': 'Washington', 'East Barre': 'Washington',
    'East Montpelier': 'Washington', 'Fayston': 'Washington',
    'Marshfield': 'Washington', 'Middlesex': 'Washington',
    'Montpelier': 'Washington', 'Moretown': 'Washington',
    'Northfield': 'Washington', 'Plainfield': 'Washington',
    'Roxbury': 'Washington', 'Waitsfield': 'Washington',
    'Warren': 'Washington', 'Waterbury': 'Washington',
    'Woodbury': 'Washington', 'Worcester': 'Washington',

    # Windham County
    'Athens': 'Windham', 'Brattleboro': 'Windham', 'Brookline': 'Windham',
    'Dover': 'Windham', 'East Dover': 'Windham',
    'Dummerston': 'Windham', 'East Dummerston': 'Windham',
    'Grafton': 'Windham', 'Guilford': 'Windham',
    'Halifax': 'Windham', 'West Halifax': 'Windham',
    'Jamaica': 'Windham', 'Londonderry': 'Windham',
    'Marlboro': 'Windham', 'Newfane': 'Windham', 'Putney': 'Windham',
    'Rockingham': 'Windham', 'Bellows Falls': 'Windham', 'Saxtons River': 'Windham',
    'Somerset': 'Windham', 'Stratton': 'Windham', 'Townshend': 'Windham',
    'Vernon': 'Windham', 'Wardsboro': 'Windham', 'Westminster': 'Windham',
    'Whitingham': 'Windham', 'Wilmington': 'Windham', 'Windham': 'Windham',

    # Windsor County
    'Andover': 'Windsor', 'Baltimore': 'Windsor', 'Barnard': 'Windsor',
    'Bethel': 'Windsor', 'Bridgewater': 'Windsor', 'Brownsville': 'Windsor',
    'Cavendish': 'Windsor', 'Proctorsville': 'Windsor',  # village in Cavendish
    'Chester': 'Windsor', 'Hartford': 'Windsor', 'White River Junction': 'Windsor',
    'Quechee': 'Windsor',  # village in Hartford
    'Hartland': 'Windsor', 'Ludlow': 'Windsor', 'Norwich': 'Windsor',
    'Orford': 'Windsor',  # Actually NH but appears in data due to interstate district
    'Plymouth': 'Windsor', 'Pomfret': 'Windsor', 'South Pomfret': 'Windsor',
    'Reading': 'Windsor', 'Rochester': 'Windsor', 'Royalton': 'Windsor',
    'South Royalton': 'Windsor', 'Sharon': 'Windsor', 'Springfield': 'Windsor',
    'Stockbridge': 'Windsor', 'Weathersfield': 'Windsor', 'Ascutney': 'Windsor',
    'Weston': 'Windsor', 'West Windsor': 'Windsor', 'Windsor': 'Windsor',
    'Woodstock': 'Windsor',
}

# =============================================================================
# TASK 1: AGGREGATE SCHOOL DATA TO TOWN LEVEL
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1: AGGREGATING SCHOOL DATA TO TOWN LEVEL")
print("=" * 70)

# Filter out schools with zero or missing enrollment
school_df_clean = school_df[school_df['enrollment'] > 0].copy()
print(f"Schools with enrollment > 0: {len(school_df_clean)}")

# Calculate per-town aggregations
town_school_agg = school_df_clean.groupby('town').agg(
    # Basic counts
    total_enrollment=('enrollment', 'sum'),
    total_free_lunch=('free_lunch', 'sum'),
    total_reduced_lunch=('reduced_price_lunch', 'sum'),
    total_frpl=('free_or_reduced_price_lunch', 'sum'),
    num_schools=('school_name', 'count'),

    # School size statistics
    avg_school_enrollment=('enrollment', 'mean'),
    max_school_enrollment=('enrollment', 'max'),
    min_school_enrollment=('enrollment', 'min'),

    # FRL rate statistics across schools
    mean_school_frpl_rate=('frl_pct', 'mean'),
    max_school_frpl_rate=('frl_pct', 'max'),
    min_school_frpl_rate=('frl_pct', 'min'),
    std_school_frpl_rate=('frl_pct', 'std'),

    # School level breakdown (count by level)
    num_elementary=('school_level', lambda x: (x == 1).sum()),
    num_middle=('school_level', lambda x: (x == 2).sum()),
    num_high=('school_level', lambda x: (x == 3).sum()),
    num_combined=('school_level', lambda x: (x == 4).sum()),
).reset_index()

# Calculate weighted FRPL rate (correct calculation)
town_school_agg['weighted_frpl_rate'] = (
    town_school_agg['total_frpl'] / town_school_agg['total_enrollment'] * 100
).round(2)

# Calculate free vs reduced breakdown
town_school_agg['free_lunch_rate'] = (
    town_school_agg['total_free_lunch'] / town_school_agg['total_enrollment'] * 100
).round(2)
town_school_agg['reduced_lunch_rate'] = (
    town_school_agg['total_reduced_lunch'] / town_school_agg['total_enrollment'] * 100
).round(2)

# Calculate within-town variance (set to 0 for single-school towns)
town_school_agg['school_frpl_variance'] = town_school_agg['std_school_frpl_rate'].fillna(0) ** 2
town_school_agg.loc[town_school_agg['num_schools'] == 1, 'school_frpl_variance'] = 0

# Calculate FRPL range within town
town_school_agg['school_frpl_range'] = (
    town_school_agg['max_school_frpl_rate'] - town_school_agg['min_school_frpl_rate']
)
town_school_agg.loc[town_school_agg['num_schools'] == 1, 'school_frpl_range'] = 0

# Round numeric columns
for col in town_school_agg.select_dtypes(include=[np.number]).columns:
    town_school_agg[col] = town_school_agg[col].round(2)

print(f"Towns aggregated: {len(town_school_agg)}")

# =============================================================================
# TASK 1B: AGGREGATE BY SCHOOL LEVEL
# =============================================================================
# Get FRPL rates by school level for each town

def get_level_frpl(group, level):
    """Get weighted FRPL rate for a specific school level"""
    level_schools = group[group['school_level'] == level]
    if len(level_schools) == 0 or level_schools['enrollment'].sum() == 0:
        return np.nan
    return (level_schools['free_or_reduced_price_lunch'].sum() /
            level_schools['enrollment'].sum() * 100)

level_rates = school_df_clean.groupby('town').apply(
    lambda g: pd.Series({
        'elementary_frpl_rate': get_level_frpl(g, 1),
        'middle_frpl_rate': get_level_frpl(g, 2),
        'high_frpl_rate': get_level_frpl(g, 3),
    })
).reset_index()

town_school_agg = town_school_agg.merge(level_rates, on='town', how='left')

# =============================================================================
# TASK 2: MAP TOWNS TO COUNTIES AND MERGE
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2: MAPPING TOWNS TO COUNTIES AND MERGING DATA")
print("=" * 70)

# Add county mapping
town_school_agg['county'] = town_school_agg['town'].map(TOWN_TO_COUNTY)

# Check for unmapped towns
unmapped = town_school_agg[town_school_agg['county'].isna()]['town'].tolist()
if unmapped:
    print(f"\nWARNING: {len(unmapped)} towns not mapped to counties:")
    for t in unmapped:
        print(f"  - {t}")
    print("\nThese will have missing county data.")

mapped_count = town_school_agg['county'].notna().sum()
print(f"Towns mapped to counties: {mapped_count}/{len(town_school_agg)}")

# Merge with county data
feature_matrix = town_school_agg.merge(
    county_df,
    left_on='county',
    right_on='county_name',
    how='left'
)

# Rename county columns for clarity
feature_matrix = feature_matrix.rename(columns={
    'poverty_rate_pct': 'county_poverty_rate',
    'median_household_income': 'county_median_income',
    'snap_participation_pct': 'county_snap_rate',
})

# Drop redundant columns
feature_matrix = feature_matrix.drop(columns=['county_name', 'state_fips', 'county_fips'], errors='ignore')

# =============================================================================
# DERIVED FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("CREATING DERIVED FEATURES")
print("=" * 70)

# Town vs County comparison features
feature_matrix['frpl_vs_county_poverty'] = (
    feature_matrix['weighted_frpl_rate'] - feature_matrix['county_poverty_rate']
).round(2)

# This shows if town is worse/better than county average
feature_matrix['frpl_above_county_avg'] = (
    feature_matrix['weighted_frpl_rate'] > feature_matrix['county_poverty_rate']
).astype(int)

# Economic stress indicators
feature_matrix['high_frpl_flag'] = (feature_matrix['weighted_frpl_rate'] > 50).astype(int)
feature_matrix['very_high_frpl_flag'] = (feature_matrix['weighted_frpl_rate'] > 70).astype(int)

# School system characteristics
feature_matrix['single_school_town'] = (feature_matrix['num_schools'] == 1).astype(int)
feature_matrix['multi_school_town'] = (feature_matrix['num_schools'] > 1).astype(int)

# Enrollment categories
feature_matrix['enrollment_category'] = pd.cut(
    feature_matrix['total_enrollment'],
    bins=[0, 50, 200, 500, 1000, float('inf')],
    labels=['very_small', 'small', 'medium', 'large', 'very_large']
)

# Relative income position (standardized)
county_median_income_mean = feature_matrix['county_median_income'].mean()
county_median_income_std = feature_matrix['county_median_income'].std()
feature_matrix['county_income_zscore'] = (
    (feature_matrix['county_median_income'] - county_median_income_mean) / county_median_income_std
).round(3)

# FRPL z-score (standardized)
frpl_mean = feature_matrix['weighted_frpl_rate'].mean()
frpl_std = feature_matrix['weighted_frpl_rate'].std()
feature_matrix['frpl_zscore'] = (
    (feature_matrix['weighted_frpl_rate'] - frpl_mean) / frpl_std
).round(3)

# Composite hardship score (simple average of normalized indicators)
# Higher = more hardship
feature_matrix['hardship_score'] = (
    (feature_matrix['frpl_zscore'] * 1) +  # higher FRPL = more hardship
    (-feature_matrix['county_income_zscore'] * 1)  # lower income = more hardship
).round(3)

# =============================================================================
# TASK 3: CLEAN UP AND FLAG LOW-CONFIDENCE
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3: CLEANUP AND QUALITY FLAGS")
print("=" * 70)

# Flag low-enrollment towns
feature_matrix['low_confidence_flag'] = (feature_matrix['total_enrollment'] < 50).astype(int)
low_conf_towns = feature_matrix[feature_matrix['low_confidence_flag'] == 1]['town'].tolist()

print(f"\nLow-confidence towns (enrollment < 50): {len(low_conf_towns)}")
if low_conf_towns:
    for t in low_conf_towns:
        enrollment = feature_matrix[feature_matrix['town'] == t]['total_enrollment'].values[0]
        print(f"  - {t}: {int(enrollment)} students")

# Check for missing values in key columns
key_columns = [
    'weighted_frpl_rate', 'county_poverty_rate', 'county_median_income',
    'county_snap_rate', 'school_frpl_variance'
]

print(f"\nMissing values check:")
for col in key_columns:
    missing = feature_matrix[col].isna().sum()
    if missing > 0:
        print(f"  - {col}: {missing} missing")

# Count complete rows
complete_rows = feature_matrix.dropna(subset=key_columns)
print(f"\nRows with complete data: {len(complete_rows)}/{len(feature_matrix)}")

# Reorder columns logically
column_order = [
    # Identifiers
    'town', 'county',

    # Enrollment & school structure
    'total_enrollment', 'num_schools', 'single_school_town',
    'avg_school_enrollment', 'enrollment_category',

    # School level breakdown
    'num_elementary', 'num_middle', 'num_high', 'num_combined',

    # FRPL rates
    'weighted_frpl_rate', 'free_lunch_rate', 'reduced_lunch_rate',
    'elementary_frpl_rate', 'middle_frpl_rate', 'high_frpl_rate',

    # Within-town inequality
    'school_frpl_variance', 'school_frpl_range',
    'min_school_frpl_rate', 'max_school_frpl_rate',

    # County economic indicators
    'county_poverty_rate', 'county_median_income', 'county_snap_rate',

    # Derived/comparative features
    'frpl_vs_county_poverty', 'frpl_above_county_avg',
    'county_income_zscore', 'frpl_zscore', 'hardship_score',

    # Flags
    'high_frpl_flag', 'very_high_frpl_flag', 'low_confidence_flag',
]

# Only include columns that exist
column_order = [c for c in column_order if c in feature_matrix.columns]
feature_matrix = feature_matrix[column_order]

# Sort by town
feature_matrix = feature_matrix.sort_values('town').reset_index(drop=True)

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("FINAL FEATURE MATRIX")
print("=" * 70)
print(f"\nShape: {feature_matrix.shape[0]} towns x {feature_matrix.shape[1]} features")
print(f"\nColumns ({len(feature_matrix.columns)}):")
for i, col in enumerate(feature_matrix.columns, 1):
    print(f"  {i:2}. {col}")

print("\n" + "-" * 70)
print("PREVIEW (first 15 rows, key columns):")
print("-" * 70)
preview_cols = ['town', 'county', 'total_enrollment', 'weighted_frpl_rate',
                'school_frpl_variance', 'county_median_income', 'hardship_score']
print(feature_matrix[preview_cols].head(15).to_string(index=False))

print("\n" + "-" * 70)
print("SUMMARY STATISTICS:")
print("-" * 70)
numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
summary_cols = [
    'total_enrollment', 'weighted_frpl_rate', 'school_frpl_variance',
    'county_poverty_rate', 'county_median_income', 'county_snap_rate',
    'hardship_score'
]
summary_cols = [c for c in summary_cols if c in feature_matrix.columns]

stats = feature_matrix[summary_cols].agg(['count', 'mean', 'std', 'min', 'max']).T
stats = stats.round(2)
print(stats.to_string())

print("\n" + "-" * 70)
print("TOP 10 HIGHEST HARDSHIP TOWNS:")
print("-" * 70)
top_hardship = feature_matrix.nlargest(10, 'hardship_score')[
    ['town', 'county', 'weighted_frpl_rate', 'county_median_income', 'hardship_score']
]
print(top_hardship.to_string(index=False))

print("\n" + "-" * 70)
print("TOP 10 LOWEST HARDSHIP TOWNS:")
print("-" * 70)
low_hardship = feature_matrix.nsmallest(10, 'hardship_score')[
    ['town', 'county', 'weighted_frpl_rate', 'county_median_income', 'hardship_score']
]
print(low_hardship.to_string(index=False))

# =============================================================================
# SAVE
# =============================================================================
feature_matrix.to_csv('vermont_feature_matrix.csv', index=False)
print(f"\n{'=' * 70}")
print(f"SAVED: vermont_feature_matrix.csv")
print(f"  - {feature_matrix.shape[0]} towns")
print(f"  - {feature_matrix.shape[1]} features")
print(f"  - {len(low_conf_towns)} low-confidence towns flagged")
print(f"  - {len(unmapped)} towns without county mapping")
print("=" * 70)
