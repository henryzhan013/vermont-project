"""
Step 4: K-Means Clustering on PCA Results
- Determine optimal number of clusters
- Run K-Means on PC1-PC5
- Analyze and visualize clusters
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load PCA results and original feature matrix
pca_results = pd.read_csv('vermont_pca_results.csv')
feature_matrix = pd.read_csv('vermont_feature_matrix.csv')

print(f"Towns: {len(pca_results)}")
print(f"PCA components available: PC1-PC5")

# Extract PC columns for clustering
pc_cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
X = pca_results[pc_cols].values

print(f"\nData shape for clustering: {X.shape}")
print("Data is already standardized from PCA - ready for clustering.")

# =============================================================================
# DETERMINE OPTIMAL NUMBER OF CLUSTERS
# =============================================================================
print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

# Test k from 2 to 10
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"  k={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={sil_score:.3f}")

# Find best k by silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k by silhouette score: {best_k}")

# =============================================================================
# FIGURE 1: Elbow Plot and Silhouette Scores
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1 = axes[0]
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
ax1.set_title('Elbow Method: Finding Optimal k')
ax1.set_xticks(list(k_range))

# Mark the "elbow" region
ax1.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Elbow region')
ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7)
ax1.legend()

# Silhouette scores
ax2 = axes[1]
bars = ax2.bar(k_range, silhouette_scores, color='steelblue', alpha=0.7)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score: Higher = Better Defined Clusters')
ax2.set_xticks(list(k_range))

# Highlight best k
bars[best_k - 2].set_color('green')
ax2.axhline(y=max(silhouette_scores), color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig7_optimal_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig7_optimal_k.png")

# =============================================================================
# RUN FINAL K-MEANS WITH CHOSEN K
# =============================================================================
# Use k=4 or k=5 (common choice for interpretability + silhouette)
# Let's use k=5 for richer segmentation
final_k = 5
print(f"\n" + "=" * 70)
print(f"RUNNING K-MEANS WITH k={final_k}")
print("=" * 70)

kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Add cluster labels to results
pca_results['cluster'] = clusters
feature_matrix['cluster'] = clusters

print(f"\nCluster distribution:")
for c in range(final_k):
    count = (clusters == c).sum()
    print(f"  Cluster {c}: {count} towns ({count/len(clusters)*100:.1f}%)")

# =============================================================================
# ANALYZE CLUSTER PROFILES
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTER PROFILES")
print("=" * 70)

# Key features to analyze
profile_cols = [
    'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
    'county_poverty_rate', 'county_snap_rate', 'school_frpl_variance',
    'num_schools', 'hardship_score'
]

# Calculate cluster means
cluster_profiles = feature_matrix.groupby('cluster')[profile_cols].mean().round(2)
cluster_profiles['n_towns'] = feature_matrix.groupby('cluster').size()

# Reorder columns
cluster_profiles = cluster_profiles[['n_towns'] + profile_cols]

print("\nCluster Means:")
print(cluster_profiles.to_string())

# Name clusters based on characteristics
cluster_names = {}
for c in range(final_k):
    profile = cluster_profiles.loc[c]
    frpl = profile['weighted_frpl_rate']
    enrollment = profile['total_enrollment']
    income = profile['county_median_income']

    # Naming logic
    if frpl > 45 and income < 68000:
        name = "High Hardship Rural"
    elif frpl > 40 and enrollment > 500:
        name = "Struggling Mid-Size"
    elif frpl < 25 and income > 80000:
        name = "Affluent Suburban"
    elif frpl < 30 and enrollment < 300:
        name = "Small & Stable"
    elif enrollment > 1000:
        name = "Large Urban/Suburban"
    else:
        name = "Mixed/Transitional"

    cluster_names[c] = name

# Refine names based on actual data patterns
# Sort clusters by hardship score to assign meaningful names
sorted_clusters = cluster_profiles.sort_values('hardship_score', ascending=False).index.tolist()

name_options = [
    "High Hardship - Rural Poor",
    "Elevated Hardship - Small Towns",
    "Moderate - Mixed Profile",
    "Lower Hardship - Stable Towns",
    "Low Hardship - Affluent Areas"
]

cluster_names = {c: name_options[i] for i, c in enumerate(sorted_clusters)}

print("\nCluster Names (by hardship level):")
for c in range(final_k):
    print(f"  Cluster {c}: {cluster_names[c]}")

# Add cluster names to dataframes
pca_results['cluster_name'] = pca_results['cluster'].map(cluster_names)
feature_matrix['cluster_name'] = feature_matrix['cluster'].map(cluster_names)

# =============================================================================
# FIGURE 2: Clusters in PCA Space (PC1 vs PC2)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Color palette for clusters
colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71']
cluster_colors = {c: colors[i] for i, c in enumerate(sorted_clusters)}

for c in range(final_k):
    mask = clusters == c
    ax.scatter(
        X[mask, 0], X[mask, 1],
        c=cluster_colors[c],
        s=80,
        alpha=0.7,
        label=f"C{c}: {cluster_names[c]}",
        edgecolors='black',
        linewidth=0.5
    )

# Plot cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=300, marker='X',
           edgecolors='white', linewidth=2, label='Cluster Centers', zorder=10)

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel('PC1 (37.1% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel('PC2 (25.6% variance)\n← Smaller Towns | Larger Towns →', fontsize=11)
ax.set_title(f'K-Means Clustering Results (k={final_k})\nVermont Towns Grouped by Economic Hardship Profile', fontsize=14)
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('fig8_clusters_pca_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig8_clusters_pca_space.png")

# =============================================================================
# FIGURE 3: Cluster Profiles Radar/Bar Chart
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Normalize features for comparison (0-1 scale)
profile_normalized = cluster_profiles[profile_cols].copy()
for col in profile_cols:
    min_val = profile_normalized[col].min()
    max_val = profile_normalized[col].max()
    if max_val > min_val:
        profile_normalized[col] = (profile_normalized[col] - min_val) / (max_val - min_val)

# Plot 1: FRPL Rate by Cluster
ax1 = axes[0, 0]
cluster_order = sorted_clusters
frpl_vals = [cluster_profiles.loc[c, 'weighted_frpl_rate'] for c in cluster_order]
bars = ax1.barh(range(final_k), frpl_vals, color=[cluster_colors[c] for c in cluster_order])
ax1.set_yticks(range(final_k))
ax1.set_yticklabels([f"C{c}: {cluster_names[c]}" for c in cluster_order], fontsize=9)
ax1.set_xlabel('Weighted FRPL Rate (%)')
ax1.set_title('Free/Reduced Lunch Rate by Cluster', fontweight='bold')
for i, v in enumerate(frpl_vals):
    ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

# Plot 2: Median Income by Cluster
ax2 = axes[0, 1]
income_vals = [cluster_profiles.loc[c, 'county_median_income'] for c in cluster_order]
bars = ax2.barh(range(final_k), income_vals, color=[cluster_colors[c] for c in cluster_order])
ax2.set_yticks(range(final_k))
ax2.set_yticklabels([f"C{c}" for c in cluster_order], fontsize=9)
ax2.set_xlabel('County Median Income ($)')
ax2.set_title('County Median Income by Cluster', fontweight='bold')
for i, v in enumerate(income_vals):
    ax2.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

# Plot 3: Average Enrollment by Cluster
ax3 = axes[1, 0]
enroll_vals = [cluster_profiles.loc[c, 'total_enrollment'] for c in cluster_order]
bars = ax3.barh(range(final_k), enroll_vals, color=[cluster_colors[c] for c in cluster_order])
ax3.set_yticks(range(final_k))
ax3.set_yticklabels([f"C{c}: {cluster_names[c]}" for c in cluster_order], fontsize=9)
ax3.set_xlabel('Average Total Enrollment')
ax3.set_title('Average Town Enrollment by Cluster', fontweight='bold')
for i, v in enumerate(enroll_vals):
    ax3.text(v + 20, i, f'{v:.0f}', va='center', fontsize=9)

# Plot 4: Number of Towns per Cluster
ax4 = axes[1, 1]
count_vals = [cluster_profiles.loc[c, 'n_towns'] for c in cluster_order]
bars = ax4.barh(range(final_k), count_vals, color=[cluster_colors[c] for c in cluster_order])
ax4.set_yticks(range(final_k))
ax4.set_yticklabels([f"C{c}" for c in cluster_order], fontsize=9)
ax4.set_xlabel('Number of Towns')
ax4.set_title('Towns per Cluster', fontweight='bold')
for i, v in enumerate(count_vals):
    ax4.text(v + 1, i, f'{int(v)}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig9_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig9_cluster_profiles.png")

# =============================================================================
# FIGURE 4: Geographic Distribution (Clusters by County)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Cross-tabulate county and cluster
county_cluster = pd.crosstab(feature_matrix['county'], feature_matrix['cluster_name'])

# Reorder columns by hardship
county_cluster = county_cluster[name_options]

# Sort counties by total hardship (weighted by cluster assignment)
county_cluster.plot(kind='barh', stacked=True, ax=ax,
                    color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71'],
                    edgecolor='white', linewidth=0.5)

ax.set_xlabel('Number of Towns')
ax.set_ylabel('County')
ax.set_title('Cluster Distribution by County\n(Which counties have the most hardship?)', fontsize=14)
ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('fig10_clusters_by_county.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig10_clusters_by_county.png")

# =============================================================================
# FIGURE 5: Example Towns from Each Cluster
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Get representative towns from each cluster (closest to centroid)
from scipy.spatial.distance import cdist

representative_towns = []
for c in range(final_k):
    cluster_mask = clusters == c
    cluster_points = X[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]

    # Find point closest to centroid
    distances = cdist([kmeans.cluster_centers_[c]], cluster_points)[0]
    closest_idx = cluster_indices[np.argmin(distances)]
    representative_towns.append(closest_idx)

# Also get extreme examples
all_examples = []
for c in range(final_k):
    cluster_data = feature_matrix[feature_matrix['cluster'] == c].nsmallest(3, 'hardship_score')
    cluster_data = pd.concat([cluster_data,
                              feature_matrix[feature_matrix['cluster'] == c].nlargest(3, 'hardship_score')])
    all_examples.append(cluster_data[['town', 'county', 'cluster_name', 'weighted_frpl_rate',
                                       'county_median_income', 'total_enrollment']].drop_duplicates())

# Create text summary for the figure
ax.axis('off')
y_pos = 0.95

ax.text(0.5, y_pos, 'Sample Towns from Each Cluster', fontsize=16, fontweight='bold',
        ha='center', transform=ax.transAxes)
y_pos -= 0.05

for c in sorted_clusters:
    cluster_towns = feature_matrix[feature_matrix['cluster'] == c].sort_values('weighted_frpl_rate', ascending=False)
    sample = cluster_towns.head(5)[['town', 'county', 'weighted_frpl_rate']].values

    y_pos -= 0.03
    ax.text(0.05, y_pos, f"Cluster {c}: {cluster_names[c]}", fontsize=12, fontweight='bold',
            transform=ax.transAxes, color=cluster_colors[c])
    y_pos -= 0.02

    towns_str = ", ".join([f"{t[0]} ({t[2]:.0f}%)" for t in sample])
    ax.text(0.08, y_pos, towns_str, fontsize=10, transform=ax.transAxes)
    y_pos -= 0.04

plt.savefig('fig11_sample_towns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig11_sample_towns.png")

# =============================================================================
# FIGURE 6: Cluster Summary Heatmap
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Normalize cluster profiles for heatmap
heatmap_data = cluster_profiles[profile_cols].copy()
for col in profile_cols:
    mean = heatmap_data[col].mean()
    std = heatmap_data[col].std()
    if std > 0:
        heatmap_data[col] = (heatmap_data[col] - mean) / std

# Reorder by hardship
heatmap_data = heatmap_data.loc[sorted_clusters]
heatmap_data.index = [f"C{c}: {cluster_names[c]}" for c in sorted_clusters]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            linewidths=0.5, ax=ax, annot_kws={'size': 10},
            cbar_kws={'label': 'Standardized Value (z-score)'})

ax.set_title('Cluster Profile Comparison\n(Red = Higher than average, Green = Lower than average)', fontsize=14)
ax.set_xlabel('Feature')
ax.set_ylabel('Cluster')

plt.tight_layout()
plt.savefig('fig12_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig12_cluster_heatmap.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Full results with cluster assignments
output_cols = ['town', 'county', 'cluster', 'cluster_name',
               'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
               'hardship_score', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# Merge to get all columns
final_output = pca_results[['town', 'county', 'cluster', 'cluster_name',
                             'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
                             'hardship_score', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

final_output = final_output.sort_values(['cluster', 'hardship_score'], ascending=[True, False])
final_output.to_csv('vermont_clusters.csv', index=False)
print(f"Saved: vermont_clusters.csv ({len(final_output)} towns)")

# Cluster summary
cluster_summary = cluster_profiles.copy()
cluster_summary['cluster_name'] = [cluster_names[c] for c in cluster_summary.index]
cluster_summary = cluster_summary[['cluster_name', 'n_towns'] + profile_cols]
cluster_summary.to_csv('vermont_cluster_summary.csv')
print("Saved: vermont_cluster_summary.csv")

# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTERING COMPLETE")
print("=" * 70)

print(f"""
K-MEANS CLUSTERING RESULTS (k={final_k})

CLUSTER SUMMARY:
""")

for c in sorted_clusters:
    profile = cluster_profiles.loc[c]
    towns = feature_matrix[feature_matrix['cluster'] == c]['town'].tolist()
    print(f"CLUSTER {c}: {cluster_names[c]}")
    print(f"  Towns: {int(profile['n_towns'])}")
    print(f"  Avg FRPL Rate: {profile['weighted_frpl_rate']:.1f}%")
    print(f"  Avg Enrollment: {profile['total_enrollment']:.0f}")
    print(f"  Avg County Income: ${profile['county_median_income']:,.0f}")
    print(f"  Example towns: {', '.join(towns[:5])}")
    print()

print("""
FILES SAVED:
  - vermont_clusters.csv (all towns with cluster assignments)
  - vermont_cluster_summary.csv (cluster profiles)
  - fig7_optimal_k.png (elbow & silhouette plots)
  - fig8_clusters_pca_space.png (clusters in PC1-PC2 space)
  - fig9_cluster_profiles.png (cluster characteristics)
  - fig10_clusters_by_county.png (geographic distribution)
  - fig11_sample_towns.png (example towns per cluster)
  - fig12_cluster_heatmap.png (cluster comparison heatmap)
""")
