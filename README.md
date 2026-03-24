# Vermont Economic Hardship Analysis

Unsupervised learning analysis of economic hardship across Vermont's 191 towns using K-Means clustering, PCA, and anomaly detection.

**Data**: Census ACS 2022 (county economics) + NCES CCD 2022 (school lunch eligibility)

---

## PCA Results

Reduced 32 features to 5 principal components (84% variance explained).

![PCA Scatter](fig3_pca_scatter.png)

- **PC1 (37%)**: Economic hardship — higher values = more struggling
- **PC2 (26%)**: Town size — larger towns at top, small rural at bottom

---

## K-Means Clustering

![Clusters](fig8_clusters_pca_space.png)

| Cluster | Name | Towns | Avg FRPL |
|---------|------|-------|----------|
| 1 (Red) | High Hardship - Rural Poor | 35 | 54% |
| 2 (Orange) | Elevated Hardship | 45 | 28% |
| 3 (Purple) | Moderate | 47 | 39% |
| 4 (Blue) | Urban Centers | 15 | 31% |
| 5 (Green) | Affluent Areas | 49 | 17% |

---

## Geographic Distribution

![Vermont Map](fig17_vermont_map.png)

- **Northeast Kingdom** (Orleans, Essex): Mostly red/orange — highest hardship
- **Chittenden County**: Green — most affluent
- **Alburgh**: 67.5% FRPL in Grand Isle County (Vermont's wealthiest) — a "hidden need" town

---

## Anomaly Detection

19 towns flagged as outliers. Key examples:

- **Alburgh**: High poverty in wealthy county
- **Burlington**: 48% FRPL, high inequality between schools
- **Newport**: 59% FRPL, regional hub in struggling area

---

## Files

- `vermont_clusters.csv` — 191 towns with cluster assignments
- `vermont_anomalies.csv` — 19 anomaly towns
- `vermont_feature_matrix.csv` — Full feature matrix
