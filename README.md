# Vermont Economic Hardship Analysis

Using public data from the U.S. Census Bureau (county-level poverty, income, SNAP rates) and NCES Common Core of Data (school-level free/reduced lunch eligibility), I analyzed economic hardship across Vermont's 191 towns. I applied PCA to reduce 32 features into interpretable components, then used K-Means clustering to group towns by hardship profile, and finally ran anomaly detection to identify towns with unusual patterns that might be overlooked.

---

## PCA Results

I reduced the 32 features down to 5 principal components that explain 84% of the variance.

![PCA Scatter](figures/fig3_pca_scatter.png)

- **PC1 (37%)**: Economic hardship — higher values = more struggling
- **PC2 (26%)**: Town size — larger towns at top, small rural at bottom

---

## K-Means Clustering

I grouped the towns into 5 clusters, ordered from most to least hardship:

![Clusters](figures/fig8_clusters_pca_space.png)

| Cluster | Name | Towns | Avg FRPL |
|---------|------|-------|----------|
| 1 (Red) | High Hardship - Rural Poor | 35 | 54% |
| 2 (Orange) | Elevated Hardship | 45 | 28% |
| 3 (Purple) | Moderate | 47 | 39% |
| 4 (Blue) | Urban Centers | 15 | 31% |
| 5 (Green) | Affluent Areas | 49 | 17% |

---

## Geographic Distribution

![Vermont Map](figures/fig17_vermont_map.png)

- **Northeast Kingdom** (Orleans, Essex, Caledonia): Mostly red/orange — highest hardship
- **Chittenden County**: Mostly green — most affluent

---

## Anomaly Detection

I used Isolation Forest to find towns that don't fit the usual patterns. One stood out:

- **Alburgh** (Grand Isle County): 67.5% FRPL in one of Vermont's wealthiest counties — a struggling community that might get overlooked because the county-level stats look fine.
