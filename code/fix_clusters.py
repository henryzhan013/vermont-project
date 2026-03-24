"""
Fix cluster labels (1-5, worst to best) and use more distinct colors
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Load data
pca_results = pd.read_csv('vermont_pca_results.csv')
feature_matrix = pd.read_csv('vermont_feature_matrix.csv')

# Get PC columns
pc_cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
X = pca_results[pc_cols].values

# Run K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X)

# Add raw clusters temporarily
feature_matrix['raw_cluster'] = raw_clusters

# Calculate mean hardship score per cluster
cluster_hardship = feature_matrix.groupby('raw_cluster')['hardship_score'].mean().sort_values(ascending=False)

# Create mapping: highest hardship = 1, lowest = 5
old_to_new = {old: new for new, old in enumerate(cluster_hardship.index, start=1)}

print("Cluster remapping (by hardship, worst to best):")
for old, new in sorted(old_to_new.items(), key=lambda x: x[1]):
    avg_hardship = cluster_hardship[old]
    print(f"  Old cluster {old} -> New cluster {new} (hardship score: {avg_hardship:.2f})")

# Apply new labels
feature_matrix['cluster'] = feature_matrix['raw_cluster'].map(old_to_new)
pca_results['cluster'] = pd.Series(raw_clusters).map(old_to_new).values

# Remap cluster centers too
new_centers = np.zeros_like(kmeans.cluster_centers_)
for old, new in old_to_new.items():
    new_centers[new-1] = kmeans.cluster_centers_[old]

# Define cluster names (1 = worst, 5 = best)
cluster_names = {
    1: "High Hardship - Rural Poor",
    2: "Elevated Hardship - Small Towns",
    3: "Moderate - Mixed Profile",
    4: "Lower Hardship - Urban Centers",
    5: "Low Hardship - Affluent Areas"
}

feature_matrix['cluster_name'] = feature_matrix['cluster'].map(cluster_names)
pca_results['cluster_name'] = pca_results['cluster'].map(cluster_names)

# Drop temporary column
feature_matrix = feature_matrix.drop(columns=['raw_cluster'])

# =============================================================================
# DISTINCT COLOR PALETTE
# =============================================================================
# Using colors that are easy to distinguish, even for colorblind viewers
colors = {
    1: '#d62728',  # Red - High hardship
    2: '#ff7f0e',  # Orange - Elevated hardship
    3: '#9467bd',  # Purple - Moderate (distinct from green/yellow)
    4: '#1f77b4',  # Blue - Lower hardship
    5: '#2ca02c',  # Green - Low hardship (affluent)
}

# =============================================================================
# FIGURE 8: Clusters in PCA Space (FIXED)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

for c in range(1, 6):
    mask = pca_results['cluster'] == c
    ax.scatter(
        X[mask, 0], X[mask, 1],
        c=colors[c],
        s=80,
        alpha=0.7,
        label=f"{c}: {cluster_names[c]}",
        edgecolors='black',
        linewidth=0.5
    )

# Plot cluster centers
ax.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=300, marker='X',
           edgecolors='white', linewidth=2, label='Cluster Centers', zorder=10)

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel('PC1 (37.1% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel('PC2 (25.6% variance)\n← Smaller Towns | Larger Towns →', fontsize=11)
ax.set_title('K-Means Clustering Results (k=5)\nVermont Towns Grouped by Economic Hardship Profile', fontsize=14)
ax.legend(loc='upper left', fontsize=9, title='Cluster (1=Worst, 5=Best)')

plt.tight_layout()
plt.savefig('fig8_clusters_pca_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig8_clusters_pca_space.png")

# =============================================================================
# FIGURE 9: Cluster Profiles (FIXED)
# =============================================================================
profile_cols = ['weighted_frpl_rate', 'total_enrollment', 'county_median_income',
                'county_poverty_rate', 'county_snap_rate', 'school_frpl_variance',
                'num_schools', 'hardship_score']

cluster_profiles = feature_matrix.groupby('cluster')[profile_cols].mean().round(2)
cluster_profiles['n_towns'] = feature_matrix.groupby('cluster').size()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

cluster_order = [1, 2, 3, 4, 5]  # Worst to best

# Plot 1: FRPL Rate by Cluster
ax1 = axes[0, 0]
frpl_vals = [cluster_profiles.loc[c, 'weighted_frpl_rate'] for c in cluster_order]
bars = ax1.barh(range(5), frpl_vals, color=[colors[c] for c in cluster_order])
ax1.set_yticks(range(5))
ax1.set_yticklabels([f"{c}: {cluster_names[c]}" for c in cluster_order], fontsize=9)
ax1.set_xlabel('Weighted FRPL Rate (%)')
ax1.set_title('Free/Reduced Lunch Rate by Cluster', fontweight='bold')
for i, v in enumerate(frpl_vals):
    ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

# Plot 2: Median Income by Cluster
ax2 = axes[0, 1]
income_vals = [cluster_profiles.loc[c, 'county_median_income'] for c in cluster_order]
bars = ax2.barh(range(5), income_vals, color=[colors[c] for c in cluster_order])
ax2.set_yticks(range(5))
ax2.set_yticklabels([f"{c}" for c in cluster_order], fontsize=9)
ax2.set_xlabel('County Median Income ($)')
ax2.set_title('County Median Income by Cluster', fontweight='bold')
for i, v in enumerate(income_vals):
    ax2.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

# Plot 3: Average Enrollment by Cluster
ax3 = axes[1, 0]
enroll_vals = [cluster_profiles.loc[c, 'total_enrollment'] for c in cluster_order]
bars = ax3.barh(range(5), enroll_vals, color=[colors[c] for c in cluster_order])
ax3.set_yticks(range(5))
ax3.set_yticklabels([f"{c}: {cluster_names[c]}" for c in cluster_order], fontsize=9)
ax3.set_xlabel('Average Total Enrollment')
ax3.set_title('Average Town Enrollment by Cluster', fontweight='bold')
for i, v in enumerate(enroll_vals):
    ax3.text(v + 20, i, f'{v:.0f}', va='center', fontsize=9)

# Plot 4: Number of Towns per Cluster
ax4 = axes[1, 1]
count_vals = [cluster_profiles.loc[c, 'n_towns'] for c in cluster_order]
bars = ax4.barh(range(5), count_vals, color=[colors[c] for c in cluster_order])
ax4.set_yticks(range(5))
ax4.set_yticklabels([f"{c}" for c in cluster_order], fontsize=9)
ax4.set_xlabel('Number of Towns')
ax4.set_title('Towns per Cluster', fontweight='bold')
for i, v in enumerate(count_vals):
    ax4.text(v + 1, i, f'{int(v)}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig9_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig9_cluster_profiles.png")

# =============================================================================
# FIGURE 10: Geographic Distribution (FIXED)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

county_cluster = pd.crosstab(feature_matrix['county'], feature_matrix['cluster'])
county_cluster = county_cluster[[1, 2, 3, 4, 5]]  # Order columns

county_cluster.plot(kind='barh', stacked=True, ax=ax,
                    color=[colors[c] for c in [1, 2, 3, 4, 5]],
                    edgecolor='white', linewidth=0.5)

ax.set_xlabel('Number of Towns')
ax.set_ylabel('County')
ax.set_title('Cluster Distribution by County\n(Which counties have the most hardship?)', fontsize=14)
ax.legend(title='Cluster', labels=[f"{c}: {cluster_names[c]}" for c in [1,2,3,4,5]],
          bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('fig10_clusters_by_county.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig10_clusters_by_county.png")

# =============================================================================
# FIGURE 12: Cluster Heatmap (FIXED)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

heatmap_data = cluster_profiles[profile_cols].copy()
for col in profile_cols:
    mean = heatmap_data[col].mean()
    std = heatmap_data[col].std()
    if std > 0:
        heatmap_data[col] = (heatmap_data[col] - mean) / std

heatmap_data = heatmap_data.loc[[1, 2, 3, 4, 5]]  # Order by cluster number
heatmap_data.index = [f"{c}: {cluster_names[c]}" for c in [1, 2, 3, 4, 5]]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            linewidths=0.5, ax=ax, annot_kws={'size': 10},
            cbar_kws={'label': 'Standardized Value (z-score)'})

ax.set_title('Cluster Profile Comparison\n(Red = Higher than average, Green = Lower than average)', fontsize=14)
ax.set_xlabel('Feature')
ax.set_ylabel('Cluster (1=Worst Hardship, 5=Best)')

plt.tight_layout()
plt.savefig('fig12_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig12_cluster_heatmap.png")

# =============================================================================
# SAVE UPDATED CSVs
# =============================================================================
output_cols = ['town', 'county', 'cluster', 'cluster_name',
               'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
               'hardship_score']

# Add PC columns from pca_results
for pc in pc_cols:
    feature_matrix[pc] = pca_results[pc]

final_output = feature_matrix[['town', 'county', 'cluster', 'cluster_name',
                                'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
                                'hardship_score', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

final_output = final_output.sort_values(['cluster', 'hardship_score'], ascending=[True, False])
final_output.to_csv('vermont_clusters.csv', index=False)
print("Saved: vermont_clusters.csv")

# Cluster summary
cluster_summary = cluster_profiles.copy()
cluster_summary['cluster_name'] = [cluster_names[c] for c in cluster_summary.index]
cluster_summary = cluster_summary[['cluster_name', 'n_towns'] + profile_cols]
cluster_summary.to_csv('vermont_cluster_summary.csv')
print("Saved: vermont_cluster_summary.csv")

print("\n" + "=" * 60)
print("UPDATED CLUSTER SUMMARY (1=Worst, 5=Best)")
print("=" * 60)
for c in [1, 2, 3, 4, 5]:
    p = cluster_profiles.loc[c]
    print(f"\nCluster {c}: {cluster_names[c]}")
    print(f"  Towns: {int(p['n_towns'])}")
    print(f"  Avg FRPL: {p['weighted_frpl_rate']:.1f}%")
    print(f"  Avg Income: ${p['county_median_income']:,.0f}")
