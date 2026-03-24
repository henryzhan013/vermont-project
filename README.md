# Vermont Town-Level Economic Hardship Analysis

An unsupervised learning approach to identify economic hardship patterns across Vermont's 191 towns using K-Means clustering, PCA, and anomaly detection.

## Data Sources

- **Census ACS 2022**: County-level poverty rate, median household income, SNAP participation
- **NCES Common Core of Data 2022**: School-level free/reduced lunch eligibility

## Key Findings

- **191 towns** analyzed across all 14 Vermont counties
- **35 towns (18%)** classified as "High Hardship"
- **Free/Reduced Lunch rates** range from 4% (Charlotte) to 75% (North Troy)
- **19 anomaly towns** identified with unusual hardship patterns

---

## Principal Component Analysis (PCA)

We reduced 32 economic features down to 5 principal components that capture 84% of the variance.

![PCA Scatter Plot](fig3_pca_scatter.png)

### What the Components Represent

| Component | Variance | Interpretation |
|-----------|----------|----------------|
| **PC1** | 37% | **Economic Hardship** — High FRPL rates, high poverty, low income, high SNAP participation. Towns on the right side of the plot face more economic challenges. |
| **PC2** | 26% | **Town Size** — Total enrollment, number of schools, within-town variance. Towns at the top are larger (Burlington, Rutland), towns at the bottom are small rural communities. |
| **PC3** | 12% | **County vs Town Effects** — Captures differences between town-level and county-level indicators. |
| **PC4** | 6% | **School Structure** — Whether a town has separate elementary/middle/high schools vs. combined schools. |
| **PC5** | 4% | **Within-Town Inequality** — Variance in FRPL rates across different schools in the same town. |

**Key insight**: The horizontal axis (PC1) essentially ranks towns from most affluent (left) to most struggling (right). The vertical axis (PC2) separates large urban centers from small rural towns.

---

## K-Means Clustering

We grouped towns into 5 clusters based on their economic profiles.

![Clusters in PCA Space](fig8_clusters_pca_space.png)

### Cluster Definitions

| Cluster | Name | Towns | Avg FRPL | Description |
|---------|------|-------|----------|-------------|
| **1** (Red) | High Hardship - Rural Poor | 35 | 54% | Small rural towns with severe economic stress. Concentrated in Orleans and Essex counties (the "Northeast Kingdom"). |
| **2** (Orange) | Elevated Hardship - Small Towns | 45 | 28% | Small towns in lower-income counties. Not as severe as Cluster 1, but still facing challenges. |
| **3** (Purple) | Moderate - Mixed Profile | 47 | 39% | Middle-of-the-road towns. Mix of economic indicators, often in transitional areas. |
| **4** (Blue) | Lower Hardship - Urban Centers | 15 | 31% | Larger towns and small cities (Burlington, Rutland, Bennington). Higher FRPL than Cluster 5 due to urban poverty pockets. |
| **5** (Green) | Low Hardship - Affluent Areas | 49 | 17% | Wealthy suburban and resort towns. Includes Stowe, Woodstock, Charlotte, and Burlington suburbs. |

---

## Geographic Distribution

![Vermont Map by Cluster](fig17_vermont_map.png)

The map reveals clear geographic patterns:

- **Northeast Kingdom** (Orleans, Essex, Caledonia counties): Dominated by red and orange — this is Vermont's economically struggling region
- **Chittenden County** (Burlington area): Almost entirely green — the state's economic engine
- **Southern Vermont**: Mixed, with pockets of hardship in Bennington and Windham counties
- **Resort corridor** (Stowe, Manchester, Woodstock): Green islands of affluence

---

## Anomaly Detection: Hidden Need Towns

Using Isolation Forest, we identified 19 towns that don't fit typical patterns. These "hidden need" cases are particularly important for policy because they might be overlooked by standard analysis.

### Notable Anomalies

**Alburgh** (Grand Isle County)
- 67.5% FRPL rate — one of the highest in the state
- Located in Grand Isle County, which has the *highest* median income ($86,639) in Vermont
- This town would be completely missed by county-level analysis
- **Why it matters**: A struggling community hidden within a wealthy county

**Burlington** (Chittenden County)
- 48% FRPL rate despite being in Vermont's wealthiest county
- High within-town inequality: schools range from 20% to 70% FRPL
- **Why it matters**: Urban poverty concentrated in specific neighborhoods

**Newport** (Orleans County)
- 59% FRPL rate with high school-to-school variance
- Largest town in the Northeast Kingdom
- **Why it matters**: Regional hub that could anchor economic development efforts

**Plainfield** (Washington County)
- 33% FRPL rate, but extremely high variance between schools
- **Why it matters**: Hidden inequality within an otherwise moderate town

---

## Files

| File | Description |
|------|-------------|
| `vermont_clusters.csv` | All 191 towns with cluster assignments and key metrics |
| `vermont_anomalies.csv` | Detailed analysis of 19 anomaly towns |
| `vermont_feature_matrix.csv` | Full 32-feature matrix used for analysis |
| `fig3_pca_scatter.png` | Towns in PCA space colored by FRPL rate |
| `fig8_clusters_pca_space.png` | Towns colored by cluster assignment |
| `fig17_vermont_map.png` | Geographic map of Vermont by cluster |

## Methodology

1. **Data Collection**: Census ACS API + Urban Institute Education Data API
2. **Feature Engineering**: 32 variables including weighted FRPL rates, within-town variance, county economics
3. **PCA**: Reduced to 5 components (84% variance explained)
4. **K-Means**: k=5 clusters determined by silhouette score
5. **Anomaly Detection**: Isolation Forest with 10% contamination threshold

## Tools

Python, pandas, scikit-learn, geopandas, matplotlib

---

*Analysis conducted March 2026*
