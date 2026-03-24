"""
Create Vermont Town Map Colored by Cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try to import geopandas, if not available we'll create a schematic map
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("geopandas not installed - will create schematic map")

# Load cluster data
clusters_df = pd.read_csv('vermont_clusters.csv')
anomalies_df = pd.read_csv('vermont_anomalies.csv')

print(f"Towns with cluster data: {len(clusters_df)}")

# Cluster colors (matching our scheme)
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

# =============================================================================
# TRY TO GET VERMONT SHAPEFILE
# =============================================================================

if HAS_GEOPANDAS:
    # Try to download Vermont towns shapefile from Census
    try:
        # Census TIGER/Line shapefiles URL for Vermont county subdivisions (towns)
        vt_url = "https://www2.census.gov/geo/tiger/TIGER2022/COUSUB/tl_2022_50_cousub.zip"

        print("Downloading Vermont town boundaries...")
        vt_towns = gpd.read_file(vt_url)
        print(f"Downloaded {len(vt_towns)} geographic units")

        # Clean town names for matching
        vt_towns['town_clean'] = vt_towns['NAME'].str.strip()
        clusters_df['town_clean'] = clusters_df['town'].str.strip()

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

        MAP_CREATED = True

    except Exception as e:
        print(f"Could not download shapefile: {e}")
        HAS_GEOPANDAS = False
        MAP_CREATED = False

# =============================================================================
# FALLBACK: Create a schematic county-based visualization
# =============================================================================

if not HAS_GEOPANDAS or not MAP_CREATED:
    print("\nCreating county-level summary map...")

    # Vermont counties approximate positions (for schematic)
    # These are rough centroid positions for visualization
    county_positions = {
        'Grand Isle': (0.3, 0.95),
        'Franklin': (0.5, 0.88),
        'Orleans': (0.7, 0.85),
        'Essex': (0.9, 0.80),
        'Lamoille': (0.5, 0.75),
        'Caledonia': (0.8, 0.70),
        'Chittenden': (0.3, 0.68),
        'Washington': (0.5, 0.60),
        'Addison': (0.25, 0.50),
        'Orange': (0.7, 0.50),
        'Rutland': (0.3, 0.35),
        'Windsor': (0.6, 0.35),
        'Bennington': (0.25, 0.15),
        'Windham': (0.55, 0.15),
    }

    # Calculate dominant cluster per county
    county_clusters = clusters_df.groupby('county')['cluster'].agg(lambda x: x.mode()[0]).to_dict()
    county_counts = clusters_df.groupby('county').size().to_dict()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    # LEFT: Schematic county map
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    for county, (x, y) in county_positions.items():
        cluster = county_clusters.get(county, 3)
        color = cluster_colors.get(cluster, 'gray')
        count = county_counts.get(county, 0)

        # Draw county as circle
        circle = plt.Circle((x, y), 0.08, color=color, alpha=0.8, ec='black', linewidth=1)
        ax1.add_patch(circle)

        # Add county name
        ax1.text(x, y + 0.01, county, ha='center', va='center', fontsize=9, fontweight='bold')
        ax1.text(x, y - 0.03, f'({count} towns)', ha='center', va='center', fontsize=7)

    # Add legend
    legend_patches = [mpatches.Patch(color=cluster_colors[c], label=f"{c}: {cluster_names[c]}")
                     for c in [1, 2, 3, 4, 5]]
    ax1.legend(handles=legend_patches, loc='lower left', fontsize=8, title='Dominant Cluster')

    ax1.set_title('Vermont Counties by Dominant Hardship Cluster\n(Schematic View)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Add Vermont outline shape approximation
    vt_outline_x = [0.1, 0.2, 0.3, 0.5, 0.6, 0.95, 0.9, 0.85, 0.7, 0.5, 0.4, 0.2, 0.1]
    vt_outline_y = [0.3, 0.05, 0.05, 0.08, 0.05, 0.7, 0.85, 0.95, 0.95, 0.92, 0.85, 0.6, 0.3]
    ax1.plot(vt_outline_x, vt_outline_y, 'k-', alpha=0.3, linewidth=2)

    # RIGHT: Detailed breakdown by county
    ax2 = axes[1]

    # Stacked bar chart showing cluster composition per county
    counties = list(county_positions.keys())

    # Calculate cluster counts per county
    cluster_by_county = pd.crosstab(clusters_df['county'], clusters_df['cluster'])
    cluster_by_county = cluster_by_county.reindex(counties)
    cluster_by_county = cluster_by_county.fillna(0)

    # Plot stacked bars
    bottom = np.zeros(len(counties))
    for cluster in [1, 2, 3, 4, 5]:
        if cluster in cluster_by_county.columns:
            values = cluster_by_county[cluster].values
            ax2.barh(counties, values, left=bottom, color=cluster_colors[cluster],
                    label=f"{cluster}: {cluster_names[cluster]}", edgecolor='white', linewidth=0.5)
            bottom += values

    ax2.set_xlabel('Number of Towns')
    ax2.set_title('Cluster Distribution by County', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8, title='Cluster')

    plt.tight_layout()
    plt.savefig('fig17_vermont_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig17_vermont_map.png")

# =============================================================================
# Create a comprehensive summary figure with all key visualizations
# =============================================================================

print("\nCreating comprehensive summary figure...")

fig = plt.figure(figsize=(20, 16))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Cluster distribution pie chart
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = clusters_df['cluster'].value_counts().sort_index()
colors = [cluster_colors[c] for c in cluster_counts.index]
wedges, texts, autotexts = ax1.pie(cluster_counts.values, colors=colors, autopct='%1.0f%%',
                                    startangle=90, pctdistance=0.75)
ax1.set_title('Town Distribution by Cluster', fontsize=12, fontweight='bold')

# Add legend
legend_labels = [f"C{c}: {cluster_names[c][:20]}..." for c in cluster_counts.index]
ax1.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=8)

# 2. FRPL Rate by Cluster (box plot)
ax2 = fig.add_subplot(gs[0, 1])
box_data = [clusters_df[clusters_df['cluster'] == c]['weighted_frpl_rate'].values for c in [1,2,3,4,5]]
bp = ax2.boxplot(box_data, patch_artist=True)
for patch, color in zip(bp['boxes'], [cluster_colors[c] for c in [1,2,3,4,5]]):
    patch.set_facecolor(color)
ax2.set_xticklabels([f'C{c}' for c in [1,2,3,4,5]])
ax2.set_ylabel('FRPL Rate (%)')
ax2.set_title('FRPL Distribution by Cluster', fontsize=12, fontweight='bold')

# 3. County Median Income by Cluster (box plot)
ax3 = fig.add_subplot(gs[0, 2])
box_data = [clusters_df[clusters_df['cluster'] == c]['county_median_income'].values / 1000 for c in [1,2,3,4,5]]
bp = ax3.boxplot(box_data, patch_artist=True)
for patch, color in zip(bp['boxes'], [cluster_colors[c] for c in [1,2,3,4,5]]):
    patch.set_facecolor(color)
ax3.set_xticklabels([f'C{c}' for c in [1,2,3,4,5]])
ax3.set_ylabel('County Median Income ($K)')
ax3.set_title('Income Distribution by Cluster', fontsize=12, fontweight='bold')

# 4. PCA Scatter (PC1 vs PC2) - Large
ax4 = fig.add_subplot(gs[1, :2])
for c in [1, 2, 3, 4, 5]:
    mask = clusters_df['cluster'] == c
    ax4.scatter(clusters_df.loc[mask, 'PC1'], clusters_df.loc[mask, 'PC2'],
               c=cluster_colors[c], s=60, alpha=0.7, label=f"{c}: {cluster_names[c]}",
               edgecolors='black', linewidth=0.3)

ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax4.set_xlabel('PC1 (37.1% variance) - Economic Hardship →', fontsize=10)
ax4.set_ylabel('PC2 (25.6% variance) - Town Size →', fontsize=10)
ax4.set_title('Towns in Principal Component Space', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8)

# 5. Top Anomalies callout
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
ax5.set_title('Top Anomaly Towns\n(Hidden Need Cases)', fontsize=12, fontweight='bold')

y_pos = 0.95
anomaly_text = anomalies_df.sort_values('anomaly_score').head(8)
for _, row in anomaly_text.iterrows():
    color = cluster_colors.get(row['cluster'], 'gray')
    ax5.text(0.05, y_pos, f"◆ {row['town']}", fontsize=10, fontweight='bold',
            transform=ax5.transAxes, color=color)
    ax5.text(0.05, y_pos-0.06, f"   {row['frpl_rate']:.0f}% FRPL | {row['category']}",
            fontsize=8, transform=ax5.transAxes, color='gray')
    y_pos -= 0.12

# 6. Cluster Profile Comparison (bar chart)
ax6 = fig.add_subplot(gs[2, :])

# Data for comparison
metrics = ['FRPL Rate (%)', 'Enrollment (÷10)', 'Income ($K)', 'SNAP Rate (%)']
cluster_means = clusters_df.groupby('cluster').agg({
    'weighted_frpl_rate': 'mean',
    'total_enrollment': 'mean',
    'county_median_income': 'mean',
}).round(1)

x = np.arange(len(metrics))
width = 0.15

for i, c in enumerate([1, 2, 3, 4, 5]):
    vals = [
        cluster_means.loc[c, 'weighted_frpl_rate'],
        cluster_means.loc[c, 'total_enrollment'] / 10,
        cluster_means.loc[c, 'county_median_income'] / 1000,
        clusters_df[clusters_df['cluster'] == c]['county_snap_rate'].mean() if 'county_snap_rate' in clusters_df.columns else 10
    ]
    ax6.bar(x + i*width, vals, width, color=cluster_colors[c], label=f"C{c}", edgecolor='black', linewidth=0.5)

ax6.set_xticks(x + width*2)
ax6.set_xticklabels(metrics)
ax6.set_ylabel('Value')
ax6.set_title('Cluster Profile Comparison', fontsize=12, fontweight='bold')
ax6.legend(title='Cluster')

plt.suptitle('Vermont Economic Hardship Analysis - Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('fig18_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig18_summary_dashboard.png")

print("\nDone! Key visualizations created.")
