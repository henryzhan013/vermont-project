"""
Create Vermont Town Map with County Boundaries
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

clusters_df = pd.read_csv('vermont_clusters.csv')
clusters_df['town_clean'] = clusters_df['town'].str.strip()

# Fix spelling differences and map villages to their parent towns
name_fixes = {
    'Alburg': 'Alburgh',
    'Saint Albans': 'St. Albans',
    'Saint Johnsbury': 'St. Johnsbury',
    'Island Pond': 'Brighton',
    'White River Junction': 'Hartford',
    'Bellows Falls': 'Rockingham',
    'Essex Junction': 'Essex',
    'Morrisville': 'Morristown',
    'Lyndonville': 'Lyndon',
    'Enosburg Falls': 'Enosburgh',
    'Derby Line': 'Derby',
    'Newport Center': 'Newport',
    'Jeffersonville': 'Cambridge',
    'Quechee': 'Hartford',
    'Proctorsville': 'Cavendish',
    'Saxtons River': 'Rockingham',
    'North Troy': 'Troy',
    'Highgate Center': 'Highgate',
    'Montgomery Center': 'Montgomery',
    'Manchester Center': 'Manchester',
    'Underhill Center': 'Underhill',
    'East Barre': 'Barre',
    'West Charleston': 'Charleston',
    'West Burke': 'Burke',
    'Wells River': 'Newbury',
    'Craftsbury Common': 'Craftsbury',
    'Lake Elmore': 'Elmore',
    'Brownsville': 'West Windsor',
    'Ascutney': 'Weathersfield',
    'South Royalton': 'Royalton',
    'South Strafford': 'Strafford',
    'South Pomfret': 'Pomfret',
    'East Corinth': 'Corinth',
    'East Dover': 'Dover',
    'East Dummerston': 'Dummerston',
    'North Clarendon': 'Clarendon',
    'West Halifax': 'Halifax',
    'West Pawlet': 'Pawlet',
    'Rutland Town': 'Rutland',
    'Gilman': 'Lunenburg',
    'Orleans': 'Barton',
    'Cuttingsville': 'Shrewsbury',
}
clusters_df['town_clean'] = clusters_df['town_clean'].replace(name_fixes)

print("Downloading Vermont town boundaries...")
vt_towns = gpd.read_file('https://www2.census.gov/geo/tiger/TIGER2022/COUSUB/tl_2022_50_cousub.zip')
vt_towns['town_clean'] = vt_towns['NAME'].str.strip()

print("Downloading Vermont county boundaries...")
vt_counties = gpd.read_file('https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip')
vt_counties = vt_counties[vt_counties['STATEFP'] == '50']  # Vermont only

vt_map = vt_towns.merge(clusters_df, left_on='town_clean', right_on='town_clean', how='left')
print(f"Matched: {vt_map['cluster'].notna().sum()} towns")

cluster_colors = {1: '#d62728', 2: '#ff7f0e', 3: '#9467bd', 4: '#1f77b4', 5: '#2ca02c'}
cluster_names = {
    1: "High Hardship - Rural Poor",
    2: "Elevated Hardship - Small Towns",
    3: "Moderate - Mixed Profile",
    4: "Lower Hardship - Urban Centers",
    5: "Low Hardship - Affluent Areas"
}

fig, ax = plt.subplots(1, 1, figsize=(12, 16))

# Plot towns without cluster data in gray
vt_map[vt_map['cluster'].isna()].plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)

# Plot towns with cluster data
for cluster in [1, 2, 3, 4, 5]:
    cluster_data = vt_map[vt_map['cluster'] == cluster]
    if len(cluster_data) > 0:
        cluster_data.plot(ax=ax, color=cluster_colors[cluster], edgecolor='white', linewidth=0.3)

# Overlay county boundaries (thick black lines)
vt_counties.boundary.plot(ax=ax, color='black', linewidth=2)

# Add county labels
for idx, row in vt_counties.iterrows():
    centroid = row.geometry.centroid
    county_name = row['NAME']
    ax.annotate(county_name, xy=(centroid.x, centroid.y), fontsize=7, ha='center', va='center',
                color='black', fontweight='bold', alpha=0.6)

# Town labels with optional offset (x_offset, y_offset)
labels = {
    'Burlington': (0, 0),
    'Rutland': (0, 0),
    'Montpelier': (0, 0),
    'Bennington': (0, 0),
    'Brattleboro': (0, 0),
    'St. Johnsbury': (0.08, -0.04),  # Move right and down to avoid Caledonia label
    'Newport': (0, 0),
    'Stowe': (0, 0),
    'Alburgh': (0, 0),
    'Middlebury': (0, 0),
}

for town_name, (x_off, y_off) in labels.items():
    town_data = vt_map[vt_map['NAME'].str.contains(town_name, case=False, na=False)]
    if len(town_data) > 0:
        centroid = town_data.geometry.centroid.iloc[0]
        ax.annotate(town_name, xy=(centroid.x + x_off, centroid.y + y_off), fontsize=8, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85))

legend_patches = [mpatches.Patch(color=cluster_colors[c], label=f"{c}: {cluster_names[c]}")
                 for c in [1, 2, 3, 4, 5]]
legend_patches.append(mpatches.Patch(color='lightgray', label='No school data'))
ax.legend(handles=legend_patches, loc='lower left', fontsize=8, title='Cluster')

ax.set_title('Vermont Towns by Economic Hardship Cluster', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('fig17_vermont_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig17_vermont_map.png")
