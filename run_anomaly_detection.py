"""
Anomaly Detection: Find "Hidden Need" Towns
Using Isolation Forest to identify towns that don't fit typical patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("ANOMALY DETECTION: Finding Hidden Need Towns")
print("=" * 70)

clusters_df = pd.read_csv('vermont_clusters.csv')
feature_matrix = pd.read_csv('vermont_feature_matrix.csv')

# Merge to get all features
df = clusters_df.merge(feature_matrix, on=['town', 'county'], suffixes=('', '_dup'))
df = df.loc[:, ~df.columns.str.endswith('_dup')]

print(f"Towns: {len(df)}")

# Features for anomaly detection (use PCA components)
pc_cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
X = df[pc_cols].values

# =============================================================================
# RUN ISOLATION FOREST
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING ISOLATION FOREST")
print("=" * 70)

# contamination = expected proportion of anomalies (start with 10%)
iso_forest = IsolationForest(
    contamination=0.10,  # Expect ~10% anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict (-1 = anomaly, 1 = normal)
anomaly_labels = iso_forest.fit_predict(X)
anomaly_scores = iso_forest.decision_function(X)  # Lower = more anomalous

df['anomaly'] = (anomaly_labels == -1).astype(int)
df['anomaly_score'] = anomaly_scores

n_anomalies = df['anomaly'].sum()
print(f"Anomalies detected: {n_anomalies} towns ({n_anomalies/len(df)*100:.1f}%)")

# =============================================================================
# ANALYZE ANOMALIES
# =============================================================================
print("\n" + "=" * 70)
print("ANOMALY ANALYSIS")
print("=" * 70)

anomalies = df[df['anomaly'] == 1].sort_values('anomaly_score')
normal = df[df['anomaly'] == 0]

print("\nAnomalous Towns (sorted by anomaly score, most unusual first):")
print("-" * 70)

key_cols = ['town', 'county', 'cluster', 'weighted_frpl_rate',
            'total_enrollment', 'county_median_income', 'anomaly_score']

print(anomalies[key_cols].to_string(index=False))

# =============================================================================
# DETERMINE WHY EACH TOWN IS ANOMALOUS
# =============================================================================
print("\n" + "=" * 70)
print("WHY ARE THESE TOWNS ANOMALOUS?")
print("=" * 70)

# Calculate z-scores for key features to identify extremes
analysis_cols = ['weighted_frpl_rate', 'total_enrollment', 'county_median_income',
                 'school_frpl_variance', 'num_schools', 'county_snap_rate',
                 'free_lunch_rate', 'reduced_lunch_rate']

# Only use columns that exist
analysis_cols = [c for c in analysis_cols if c in df.columns]

# Calculate z-scores
z_scores = pd.DataFrame()
for col in analysis_cols:
    mean = df[col].mean()
    std = df[col].std()
    z_scores[col] = (df[col] - mean) / std

z_scores['town'] = df['town']

# For each anomaly, identify extreme features
anomaly_reasons = []

for idx, row in anomalies.iterrows():
    town = row['town']
    town_z = z_scores[z_scores['town'] == town].iloc[0]

    extremes = []
    for col in analysis_cols:
        z = town_z[col]
        if abs(z) > 1.5:  # More than 1.5 std from mean
            direction = "high" if z > 0 else "low"
            extremes.append(f"{col}: {direction} (z={z:.1f})")

    reason = "; ".join(extremes) if extremes else "Unusual combination of features"
    anomaly_reasons.append({
        'town': town,
        'county': row['county'],
        'cluster': row['cluster'],
        'cluster_name': row['cluster_name'],
        'frpl_rate': row['weighted_frpl_rate'],
        'enrollment': row['total_enrollment'],
        'income': row['county_median_income'],
        'anomaly_score': row['anomaly_score'],
        'reason': reason
    })

anomaly_df = pd.DataFrame(anomaly_reasons)

print("\nDetailed Anomaly Analysis:")
print("-" * 70)
for _, row in anomaly_df.iterrows():
    print(f"\n{row['town']} ({row['county']} County)")
    print(f"  Cluster: {row['cluster']} - {row['cluster_name']}")
    print(f"  FRPL: {row['frpl_rate']:.1f}% | Enrollment: {int(row['enrollment'])} | Income: ${row['income']:,.0f}")
    print(f"  Why anomalous: {row['reason']}")

# =============================================================================
# CATEGORIZE ANOMALY TYPES
# =============================================================================
print("\n" + "=" * 70)
print("ANOMALY CATEGORIES")
print("=" * 70)

def categorize_anomaly(row, z_scores_df):
    town_z = z_scores_df[z_scores_df['town'] == row['town']].iloc[0]

    # Check various patterns
    high_frpl = town_z['weighted_frpl_rate'] > 1.5
    low_frpl = town_z['weighted_frpl_rate'] < -1.5
    high_enrollment = town_z['total_enrollment'] > 2
    low_enrollment = town_z['total_enrollment'] < -1
    high_variance = town_z.get('school_frpl_variance', 0) > 2 if 'school_frpl_variance' in town_z.index else False

    income = row['income']  # Use 'income' column from anomaly_df

    if high_enrollment:
        return "Large Urban Center"
    elif high_frpl and income > 70000:
        return "Hidden Need - High FRPL in Wealthy County"
    elif low_frpl and income < 65000:
        return "Pocket of Prosperity - Low FRPL in Poor County"
    elif high_variance:
        return "High Inequality Within Town"
    elif high_frpl:
        return "Extreme Hardship"
    else:
        return "Unusual Profile"

anomaly_df['category'] = anomaly_df.apply(lambda r: categorize_anomaly(r, z_scores), axis=1)

print("\nAnomalies by Category:")
print(anomaly_df.groupby('category').size().sort_values(ascending=False).to_string())

print("\n" + "-" * 70)
print("HIDDEN NEED TOWNS (most interesting for policy):")
print("-" * 70)
hidden_need = anomaly_df[anomaly_df['category'].str.contains('Hidden|Inequality|Unusual')]
for _, row in hidden_need.iterrows():
    print(f"  {row['town']} ({row['county']}): {row['category']}")
    print(f"    FRPL: {row['frpl_rate']:.1f}% | County Income: ${row['income']:,.0f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# FIGURE 13: Anomalies in PCA Space
fig, ax = plt.subplots(figsize=(14, 10))

# Plot normal towns
normal_mask = df['anomaly'] == 0
ax.scatter(df.loc[normal_mask, 'PC1'], df.loc[normal_mask, 'PC2'],
           c='lightgray', s=50, alpha=0.5, label='Normal Towns', edgecolors='none')

# Plot anomalies with color by category
anomaly_mask = df['anomaly'] == 1
anomaly_towns = df[anomaly_mask].merge(anomaly_df[['town', 'category']], on='town')

category_colors = {
    'Large Urban Center': '#1f77b4',
    'Hidden Need - High FRPL in Wealthy County': '#d62728',
    'Pocket of Prosperity - Low FRPL in Poor County': '#2ca02c',
    'High Inequality Within Town': '#ff7f0e',
    'Extreme Hardship': '#9467bd',
    'Unusual Profile': '#8c564b'
}

for cat in anomaly_towns['category'].unique():
    cat_mask = anomaly_towns['category'] == cat
    ax.scatter(anomaly_towns.loc[cat_mask, 'PC1'],
               anomaly_towns.loc[cat_mask, 'PC2'],
               c=category_colors.get(cat, 'red'),
               s=150, alpha=0.9, label=cat,
               edgecolors='black', linewidth=1.5, marker='D')

# Label anomalies
for _, row in anomaly_towns.iterrows():
    ax.annotate(row['town'], (row['PC1'], row['PC2']),
                fontsize=8, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel('PC1 (37.1% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel('PC2 (25.6% variance)\n← Smaller Towns | Larger Towns →', fontsize=11)
ax.set_title('Anomaly Detection: Towns That Don\'t Fit Typical Patterns\n(Diamonds = Anomalies)', fontsize=14)
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('fig13_anomalies_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig13_anomalies_pca.png")

# FIGURE 14: Anomaly Score Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of anomaly scores
ax1 = axes[0]
ax1.hist(df['anomaly_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
threshold = df[df['anomaly'] == 1]['anomaly_score'].max()
ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Anomaly Threshold ({threshold:.2f})')
ax1.set_xlabel('Anomaly Score (lower = more anomalous)')
ax1.set_ylabel('Number of Towns')
ax1.set_title('Distribution of Anomaly Scores')
ax1.legend()

# Anomalies by cluster
ax2 = axes[1]
cluster_anomaly = df.groupby('cluster')['anomaly'].sum()
cluster_total = df.groupby('cluster').size()
cluster_pct = (cluster_anomaly / cluster_total * 100).round(1)

colors = ['#d62728', '#ff7f0e', '#9467bd', '#1f77b4', '#2ca02c']
bars = ax2.bar(range(1, 6), cluster_pct.values, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(1, 6))
ax2.set_xticklabels([f'Cluster {i}' for i in range(1, 6)])
ax2.set_ylabel('% Anomalies')
ax2.set_title('Anomaly Rate by Cluster')
for i, (v, n) in enumerate(zip(cluster_pct.values, cluster_anomaly.values)):
    ax2.text(i+1, v + 0.5, f'{int(n)} towns', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig14_anomaly_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig14_anomaly_distribution.png")

# FIGURE 15: Feature Comparison - Anomalies vs Normal
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

compare_features = ['weighted_frpl_rate', 'total_enrollment', 'county_median_income',
                    'school_frpl_variance', 'num_schools', 'county_snap_rate']
compare_features = [f for f in compare_features if f in df.columns]

for idx, feat in enumerate(compare_features):
    ax = axes[idx // 3, idx % 3]

    normal_vals = df[df['anomaly'] == 0][feat]
    anomaly_vals = df[df['anomaly'] == 1][feat]

    ax.boxplot([normal_vals, anomaly_vals], labels=['Normal', 'Anomalies'])
    ax.set_title(feat.replace('_', ' ').title())
    ax.set_ylabel('Value')

plt.suptitle('Feature Distributions: Normal Towns vs Anomalies', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig15_anomaly_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig15_anomaly_features.png")

# FIGURE 16: Hidden Need Towns Summary
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create summary text
text = """
HIDDEN NEED TOWNS - ANOMALY DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These towns don't fit typical patterns and may have hidden needs:

"""

y_pos = 0.95
ax.text(0.5, y_pos, 'Hidden Need Towns - Anomaly Detection Results',
        fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
y_pos -= 0.08

for _, row in anomaly_df.sort_values('anomaly_score').iterrows():
    if y_pos < 0.1:
        break

    color = category_colors.get(row['category'], 'black')

    ax.text(0.05, y_pos, f"◆ {row['town']}", fontsize=11, fontweight='bold',
            transform=ax.transAxes, color=color)
    ax.text(0.25, y_pos, f"({row['county']} County)", fontsize=10,
            transform=ax.transAxes, color='gray')
    y_pos -= 0.04

    ax.text(0.07, y_pos, f"FRPL: {row['frpl_rate']:.1f}%  |  Enrollment: {int(row['enrollment'])}  |  County Income: ${row['income']:,.0f}",
            fontsize=9, transform=ax.transAxes)
    y_pos -= 0.03

    ax.text(0.07, y_pos, f"Category: {row['category']}", fontsize=9,
            transform=ax.transAxes, style='italic', color=color)
    y_pos -= 0.05

plt.savefig('fig16_hidden_need_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig16_hidden_need_summary.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Full results with anomaly flags
output = df[['town', 'county', 'cluster', 'cluster_name',
             'weighted_frpl_rate', 'total_enrollment', 'county_median_income',
             'anomaly', 'anomaly_score', 'PC1', 'PC2']].copy()
output = output.sort_values('anomaly_score')
output.to_csv('vermont_with_anomalies.csv', index=False)
print("Saved: vermont_with_anomalies.csv")

# Anomaly details
anomaly_df.to_csv('vermont_anomalies.csv', index=False)
print("Saved: vermont_anomalies.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("ANOMALY DETECTION COMPLETE")
print("=" * 70)

print(f"""
SUMMARY:
  - Total towns analyzed: {len(df)}
  - Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)

ANOMALY CATEGORIES:
""")
for cat, count in anomaly_df.groupby('category').size().sort_values(ascending=False).items():
    print(f"  {cat}: {count} towns")

print("""
KEY FINDINGS - "HIDDEN NEED" TOWNS:
These towns may warrant special attention because they don't fit
typical patterns. They might look fine on one metric but struggle on another.
""")

for _, row in anomaly_df.sort_values('anomaly_score').head(10).iterrows():
    print(f"  • {row['town']} ({row['county']}): {row['category']}")

print("""
FILES SAVED:
  - vermont_with_anomalies.csv (all towns with anomaly flags)
  - vermont_anomalies.csv (detailed anomaly analysis)
  - fig13_anomalies_pca.png (anomalies in PCA space)
  - fig14_anomaly_distribution.png (anomaly score distribution)
  - fig15_anomaly_features.png (feature comparison)
  - fig16_hidden_need_summary.png (summary graphic)
""")
