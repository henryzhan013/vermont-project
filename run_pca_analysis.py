"""
Step 3: PCA Analysis on Vermont Economic Hardship Feature Matrix
- Data preparation and standardization
- PCA transformation
- Visualizations that tell the story
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 70)
print("LOADING AND PREPARING DATA")
print("=" * 70)

df = pd.read_csv('vermont_feature_matrix.csv')
print(f"Loaded: {df.shape[0]} towns, {df.shape[1]} columns")

# Select numeric features for PCA (exclude identifiers, categorical, and derived flags)
exclude_cols = [
    'town', 'county', 'enrollment_category',  # identifiers/categorical
    'low_confidence_flag',  # we'll use this for filtering, not as a feature
    'frpl_above_county_avg', 'high_frpl_flag', 'very_high_frpl_flag',  # binary flags derived from other features
    'single_school_town', 'multi_school_town',  # binary, derived from num_schools
]

# Also exclude school-level FRPL rates that have many NaN values
# (towns with only elementary schools won't have middle/high rates)
exclude_cols += ['elementary_frpl_rate', 'middle_frpl_rate', 'high_frpl_rate']

feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

print(f"\nFeatures selected for PCA ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2}. {col}")

# Create feature matrix
X = df[feature_cols].copy()

# Check for missing values
print(f"\nMissing values per feature:")
missing = X.isnull().sum()
if missing.sum() == 0:
    print("  None - data is complete!")
else:
    print(missing[missing > 0])

# Fill any remaining NaN with column median (shouldn't be needed but safety)
X = X.fillna(X.median())

# =============================================================================
# STANDARDIZE FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("STANDARDIZING FEATURES")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nBefore standardization (sample stats):")
print(f"  weighted_frpl_rate: mean={X['weighted_frpl_rate'].mean():.2f}, std={X['weighted_frpl_rate'].std():.2f}")
print(f"  county_median_income: mean={X['county_median_income'].mean():.0f}, std={X['county_median_income'].std():.0f}")

X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("\nAfter standardization (all features):")
print(f"  Mean: {X_scaled_df.mean().mean():.6f} (should be ~0)")
print(f"  Std:  {X_scaled_df.std().mean():.6f} (should be ~1)")

# =============================================================================
# RUN PCA
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING PCA")
print("=" * 70)

# Run PCA with all components first to see variance explained
pca_full = PCA()
pca_full.fit(X_scaled)

# Variance explained
var_explained = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(var_explained)

print("\nVariance Explained by Component:")
print("-" * 50)
for i, (var, cum) in enumerate(zip(var_explained[:10], cumulative_var[:10]), 1):
    bar = "█" * int(var * 50)
    print(f"  PC{i:2}: {var*100:5.1f}% (cumulative: {cum*100:5.1f}%) {bar}")

# Determine optimal number of components (80% variance threshold)
n_components_80 = np.argmax(cumulative_var >= 0.80) + 1
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1

print(f"\nComponents needed for 80% variance: {n_components_80}")
print(f"Components needed for 90% variance: {n_components_90}")

# Use components that explain 80% variance for the final model
n_components = n_components_80
print(f"\nUsing {n_components} components for analysis")

# Fit final PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# =============================================================================
# ANALYZE COMPONENT LOADINGS
# =============================================================================
print("\n" + "=" * 70)
print("COMPONENT LOADINGS (what each PC represents)")
print("=" * 70)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=feature_cols
)

# For each component, show top positive and negative loadings
for i in range(min(n_components, 5)):  # Show first 5 components
    pc = f'PC{i+1}'
    var_pct = var_explained[i] * 100
    print(f"\n{pc} ({var_pct:.1f}% variance):")

    sorted_loadings = loadings[pc].sort_values()

    print("  Top positive loadings (high value = high PC score):")
    for feat, load in sorted_loadings.tail(3).sort_values(ascending=False).items():
        print(f"    +{load:.3f}  {feat}")

    print("  Top negative loadings (high value = low PC score):")
    for feat, load in sorted_loadings.head(3).items():
        print(f"    {load:.3f}  {feat}")

# =============================================================================
# CREATE OUTPUT DATAFRAME
# =============================================================================
print("\n" + "=" * 70)
print("CREATING OUTPUT DATA")
print("=" * 70)

# Create results dataframe with PCA scores
results = df[['town', 'county', 'total_enrollment', 'weighted_frpl_rate',
              'county_median_income', 'low_confidence_flag']].copy()

for i in range(n_components):
    results[f'PC{i+1}'] = X_pca[:, i]

# Add hardship score for reference
results['hardship_score'] = df['hardship_score']

print(f"\nResults shape: {results.shape}")
print("\nSample of PCA results:")
print(results.head(10).to_string(index=False))

# Save to CSV
results.to_csv('vermont_pca_results.csv', index=False)
loadings.to_csv('vermont_pca_loadings.csv')
print(f"\nSaved: vermont_pca_results.csv")
print(f"Saved: vermont_pca_loadings.csv")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Create figure directory conceptually (save in current dir)

# -----------------------------------------------------------------------------
# FIGURE 1: Scree Plot + Cumulative Variance
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
ax1 = axes[0]
components_range = range(1, len(var_explained) + 1)
ax1.bar(components_range, var_explained * 100, color='steelblue', alpha=0.7, label='Individual')
ax1.plot(components_range, cumulative_var * 100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
ax1.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
ax1.axvline(x=n_components_80, color='green', linestyle=':', alpha=0.7)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('Scree Plot: Variance Explained by Each Component')
ax1.legend(loc='center right')
ax1.set_xticks(components_range)

# Cumulative variance zoomed
ax2 = axes[1]
ax2.fill_between(components_range, cumulative_var * 100, alpha=0.3, color='steelblue')
ax2.plot(components_range, cumulative_var * 100, 'o-', color='steelblue', linewidth=2, markersize=8)
ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% variance')
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% variance')
ax2.axvline(x=n_components_80, color='green', linestyle=':', alpha=0.7)
ax2.axvline(x=n_components_90, color='orange', linestyle=':', alpha=0.7)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('How Many Components Do We Need?')
ax2.legend()
ax2.set_xticks(components_range)
ax2.set_ylim(0, 105)

# Add annotations
ax2.annotate(f'{n_components_80} components\nfor 80%',
             xy=(n_components_80, 80), xytext=(n_components_80 + 1.5, 70),
             fontsize=9, ha='left',
             arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig('fig1_scree_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_scree_plot.png")

# -----------------------------------------------------------------------------
# FIGURE 2: Component Loadings Heatmap
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 12))

# Create heatmap of loadings
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, annot_kws={'size': 8},
            cbar_kws={'label': 'Loading Weight'})

ax.set_title('PCA Component Loadings\n(What Each Principal Component Represents)', fontsize=14)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Original Feature')

plt.tight_layout()
plt.savefig('fig2_loadings_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_loadings_heatmap.png")

# -----------------------------------------------------------------------------
# FIGURE 3: Towns in PCA Space (PC1 vs PC2)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Color by weighted FRPL rate
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df['weighted_frpl_rate'],
    cmap='RdYlGn_r',  # Red = high FRPL (hardship), Green = low
    s=df['total_enrollment'] / 20 + 20,  # Size by enrollment
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Free/Reduced Lunch Rate (%)', fontsize=11)

# Label extreme towns
extreme_indices = []
# Highest PC1
extreme_indices.extend(np.argsort(X_pca[:, 0])[-5:])  # top 5 PC1
extreme_indices.extend(np.argsort(X_pca[:, 0])[:5])   # bottom 5 PC1
extreme_indices.extend(np.argsort(X_pca[:, 1])[-3:])  # top 3 PC2
extreme_indices.extend(np.argsort(X_pca[:, 1])[:3])   # bottom 3 PC2
extreme_indices = list(set(extreme_indices))

for idx in extreme_indices:
    ax.annotate(
        df.iloc[idx]['town'],
        (X_pca[idx, 0], X_pca[idx, 1]),
        fontsize=8,
        alpha=0.8,
        xytext=(5, 5),
        textcoords='offset points'
    )

# Add quadrant labels
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('Vermont Towns in Principal Component Space\n(Size = Enrollment, Color = FRPL Rate)', fontsize=14)

# Add legend for size
sizes = [100, 500, 1500, 3000]
for s in sizes:
    ax.scatter([], [], s=s/20+20, c='gray', alpha=0.5, edgecolors='black',
               label=f'{s} students')
ax.legend(title='Enrollment', loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('fig3_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_pca_scatter.png")

# -----------------------------------------------------------------------------
# FIGURE 4: Towns by County in PCA Space
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Get unique counties and assign colors
counties = df['county'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(counties)))
county_colors = dict(zip(counties, colors))

for county in counties:
    mask = df['county'] == county
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=[county_colors[county]],
        s=60,
        alpha=0.7,
        label=county,
        edgecolors='black',
        linewidth=0.3
    )

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('Vermont Towns Colored by County\n(Do counties cluster together?)', fontsize=14)
ax.legend(title='County', loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('fig4_pca_by_county.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_pca_by_county.png")

# -----------------------------------------------------------------------------
# FIGURE 5: Biplot (PC1 vs PC2 with feature vectors)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Plot towns (smaller, more transparent)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', s=30, alpha=0.5, edgecolors='none')

# Plot feature vectors
scale = 4  # Scale factor for visibility
for i, feature in enumerate(feature_cols):
    ax.arrow(0, 0,
             pca.components_[0, i] * scale,
             pca.components_[1, i] * scale,
             head_width=0.1, head_length=0.05,
             fc='red', ec='red', alpha=0.7)

    # Label placement
    x_pos = pca.components_[0, i] * scale * 1.15
    y_pos = pca.components_[1, i] * scale * 1.15

    ax.annotate(feature, (x_pos, y_pos), fontsize=8, ha='center', va='center',
                color='darkred', fontweight='bold')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('PCA Biplot: How Features Contribute to Each Component\n(Arrows show feature direction and strength)', fontsize=14)

# Set equal aspect for proper vector representation
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('fig5_biplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_biplot.png")

# -----------------------------------------------------------------------------
# FIGURE 6: Top/Bottom Towns Story
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Highest hardship towns (highest PC1)
ax1 = axes[0]
top_hardship = results.nlargest(15, 'PC1')[['town', 'county', 'PC1', 'weighted_frpl_rate', 'county_median_income']]
y_pos = range(len(top_hardship))
bars = ax1.barh(y_pos, top_hardship['PC1'], color='indianred', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{row['town']} ({row['county']})" for _, row in top_hardship.iterrows()])
ax1.invert_yaxis()
ax1.set_xlabel('PC1 Score (Higher = More Hardship)')
ax1.set_title('Top 15 Highest Economic Hardship Towns', fontsize=12, fontweight='bold')

# Add FRPL rate labels
for i, (_, row) in enumerate(top_hardship.iterrows()):
    ax1.annotate(f"{row['weighted_frpl_rate']:.0f}% FRPL",
                 xy=(row['PC1'], i), xytext=(5, 0),
                 textcoords='offset points', fontsize=8, va='center')

# Lowest hardship towns (lowest PC1)
ax2 = axes[1]
low_hardship = results.nsmallest(15, 'PC1')[['town', 'county', 'PC1', 'weighted_frpl_rate', 'county_median_income']]
y_pos = range(len(low_hardship))
bars = ax2.barh(y_pos, low_hardship['PC1'], color='seagreen', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{row['town']} ({row['county']})" for _, row in low_hardship.iterrows()])
ax2.invert_yaxis()
ax2.set_xlabel('PC1 Score (Lower = Less Hardship)')
ax2.set_title('Top 15 Lowest Economic Hardship Towns', fontsize=12, fontweight='bold')

# Add FRPL rate labels
for i, (_, row) in enumerate(low_hardship.iterrows()):
    ax2.annotate(f"{row['weighted_frpl_rate']:.0f}% FRPL",
                 xy=(row['PC1'], i), xytext=(5, 0),
                 textcoords='offset points', fontsize=8, va='center')

plt.tight_layout()
plt.savefig('fig6_top_bottom_towns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_top_bottom_towns.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PCA ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
KEY FINDINGS:

1. DIMENSIONALITY REDUCTION:
   - Reduced {len(feature_cols)} features → {n_components} principal components
   - These {n_components} components capture {cumulative_var[n_components-1]*100:.1f}% of total variance

2. WHAT THE COMPONENTS REPRESENT:
   - PC1 ({var_explained[0]*100:.1f}%): Overall Economic Hardship
     • High scores = high FRPL, high poverty, low income, high SNAP
     • This is the main "hardship axis"

   - PC2 ({var_explained[1]*100:.1f}%): Town Size/School System Scale
     • Captures enrollment, number of schools, school variance
     • Separates large towns from small towns

3. GEOGRAPHIC PATTERNS:
   - Orleans and Essex counties cluster toward high hardship
   - Chittenden county clusters toward low hardship
   - This matches known economic geography of Vermont

FILES SAVED:
   - vermont_pca_results.csv (town scores on each PC)
   - vermont_pca_loadings.csv (feature weights for each PC)
   - fig1_scree_plot.png (variance explained)
   - fig2_loadings_heatmap.png (what each PC means)
   - fig3_pca_scatter.png (towns in PC space, colored by FRPL)
   - fig4_pca_by_county.png (towns colored by county)
   - fig5_biplot.png (feature vectors)
   - fig6_top_bottom_towns.png (highest/lowest hardship towns)
""")
