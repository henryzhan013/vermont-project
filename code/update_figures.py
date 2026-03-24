"""
Update Fig 3 and Fig 4 with better PC2 axis labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Load data
df = pd.read_csv('vermont_feature_matrix.csv')
results = pd.read_csv('vermont_pca_results.csv')

# Get PCA coordinates
X_pca = results[['PC1', 'PC2']].values

# Variance explained (from previous run)
var_explained = [0.371, 0.256]

# -----------------------------------------------------------------------------
# FIGURE 3: Towns in PCA Space (PC1 vs PC2) - UPDATED
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Color by weighted FRPL rate
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df['weighted_frpl_rate'],
    cmap='RdYlGn_r',
    s=df['total_enrollment'] / 20 + 20,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Free/Reduced Lunch Rate (%)', fontsize=11)

# Label extreme towns
extreme_indices = []
extreme_indices.extend(np.argsort(X_pca[:, 0])[-5:])
extreme_indices.extend(np.argsort(X_pca[:, 0])[:5])
extreme_indices.extend(np.argsort(X_pca[:, 1])[-3:])
extreme_indices.extend(np.argsort(X_pca[:, 1])[:3])
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

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# UPDATED AXIS LABELS with meaning
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)\n← Smaller Towns | Larger Towns →', fontsize=11)
ax.set_title('Vermont Towns in Principal Component Space\n(Size = Enrollment, Color = FRPL Rate)', fontsize=14)

# Legend for size
sizes = [100, 500, 1500, 3000]
for s in sizes:
    ax.scatter([], [], s=s/20+20, c='gray', alpha=0.5, edgecolors='black',
               label=f'{s} students')
ax.legend(title='Enrollment', loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('fig3_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Updated: fig3_pca_scatter.png")

# -----------------------------------------------------------------------------
# FIGURE 4: Towns by County in PCA Space - UPDATED
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

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

# UPDATED AXIS LABELS with meaning
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)\n← Lower Hardship | Higher Hardship →', fontsize=11)
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)\n← Smaller Towns | Larger Towns →', fontsize=11)
ax.set_title('Vermont Towns Colored by County\n(Geographic clustering visible)', fontsize=14)
ax.legend(title='County', loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('fig4_pca_by_county.png', dpi=150, bbox_inches='tight')
plt.close()
print("Updated: fig4_pca_by_county.png")

print("\nDone! Both figures now have descriptive labels for both axes.")
