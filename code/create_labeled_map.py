"""
Create Vermont Town Map with Major City Labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

# Load cluster data
clusters_df = pd.read_csv('vermont_clusters.csv')

# Cluster colors
cluster_colors = {
    1: '#d62728',  # Red - High hardship
    2: '#ff7f0e',  # Orange - Elevated hardship
    3: '#9467bd',  # Purple - Moderate
    4: '#1f77b4',  # Blue - Lower hardship
    5: '#2ca02c',  # Green - Low hardship
}

cluster_names = {
    1: "High Hardship - Rural Poor",
    2: "Elevated Hardship - Small Towns",
    3: "Moderate - Mixed Profile",
    4: "Lower Hardship - Urban Centers",
    5: "Low Hardship - Affluent Areas"
}

# Major cities/towns to label
major_towns = [
    'Burlington', 'Rutland', 'Montpelier', 'Barre', 'Bennington',
    'Brattleboro', 'St. Johnsbury', 'Newport', 'Middlebury',
    'St. Albans', 'Essex Junction', 'South Burlington', 'Stowe',
    'Manchester', 'Woodstock', 'White River Junction',
    'Alburgh', 'North Troy', 'Island Pond'  # Key high-hardship towns
]

# Alternative names that might appear in the data
town_aliases = {
    'Saint Johnsbury': 'St. Johnsbury',
    'Saint Albans': 'St. Albans',
    'Hartford': 'White River Junction',  # White River Junction is in Hartford
    'Alburg': 'Alburgh',  # Different spellings
    'Brighton': 'Island Pond',  # Island Pond is in Brighton
}

print("Downloading Vermont town boundaries...")
vt_url = "https://www2.census.gov/geo/tiger/TIGER2022/COUSUB/tl_2022_50_cousub.zip"
vt_towns = gpd.read_file(vt_url)
print(f"Downloaded {len(vt_towns)} geographic units")

# Clean town names for matching
vt_towns['town_clean'] = vt_towns['NAME'].str.strip()
clusters_df['town_clean'] = clusters_df['town'].str.strip()

# Fix spelling differences between our data and Census shapefile
# Our data uses different spellings than Census TIGER files
spelling_fixes = {
    'Alburg': 'Alburgh',
}
clusters_df['town_clean'] = clusters_df['town_clean'].replace(spelling_fixes)

# Also fix Census data to match ours where needed
census_fixes = {
    'St. Johnsbury': 'Saint Johnsbury',
    'St. Albans city': 'Saint Albans',
    'St. Albans town': 'Saint Albans',
}
vt_towns['town_clean'] = vt_towns['town_clean'].replace(census_fixes)

# Merge cluster data
vt_map = vt_towns.merge(clusters_df, left_on='town_clean', right_on='town_clean', how='left')
print(f"Matched towns: {vt_map['cluster'].notna().sum()}")

# Create the map
fig, ax = plt.subplots(1, 1, figsize=(12, 16))

# Plot towns without cluster data in gray
vt_map[vt_map['cluster'].isna()].plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

# Plot towns with cluster data
for cluster in [1, 2, 3, 4, 5]:
    cluster_data = vt_map[vt_map['cluster'] == cluster]
    if len(cluster_data) > 0:
        cluster_data.plot(ax=ax, color=cluster_colors[cluster], edgecolor='white',
                          linewidth=0.5, label=f"{cluster}: {cluster_names[cluster]}")

# Add labels for major towns
labeled_count = 0
for town in major_towns:
    # Try to find the town in the map data
    town_data = vt_map[vt_map['NAME'] == town]

    # Try aliases
    if len(town_data) == 0:
        for alias, canonical in town_aliases.items():
            if canonical == town:
                town_data = vt_map[vt_map['NAME'] == alias]
                if len(town_data) > 0:
                    break

    # Try partial match
    if len(town_data) == 0:
        town_data = vt_map[vt_map['NAME'].str.contains(town, case=False, na=False)]

    if len(town_data) > 0:
        # Get centroid for label placement
        centroid = town_data.geometry.centroid.iloc[0]

        # Get cluster for this town to determine text color
        cluster = town_data['cluster'].iloc[0]

        # Add label with white background for readability
        ax.annotate(
            town,
            xy=(centroid.x, centroid.y),
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8)
        )
        labeled_count += 1

print(f"Labeled {labeled_count} major towns")

# Also try to label some towns from the clusters data directly
additional_labels = ['Stowe', 'Woodstock', 'Manchester']
for town in additional_labels:
    town_data = vt_map[vt_map['town_clean'] == town]
    if len(town_data) > 0 and town not in major_towns[:labeled_count]:
        centroid = town_data.geometry.centroid.iloc[0]
        ax.annotate(
            town,
            xy=(centroid.x, centroid.y),
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8)
        )

# Add legend
legend_patches = [mpatches.Patch(color=cluster_colors[c], label=f"{c}: {cluster_names[c]}")
                 for c in [1, 2, 3, 4, 5]]
legend_patches.append(mpatches.Patch(color='lightgray', label='No school data'))
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, title='Cluster')

ax.set_title('Vermont Towns by Economic Hardship Cluster', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('fig17_vermont_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig17_vermont_map.png")
