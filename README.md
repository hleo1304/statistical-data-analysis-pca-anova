# Statistical Data Analysis: PCA & ANOVA in Python

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-PCA-F7931E?logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-ANOVA-150458)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)

A two-part applied statistics project exploring **dimensionality reduction** and **hypothesis testing** in Python:
1. **Principal Component Analysis (PCA)** on the **OECD Better Life Index 2024** dataset ($n = 38$ countries, $p = 25$ socioeconomic and well-being indicators), including an outlier sensitivity analysis.
2. **One-Way & Two-Way ANOVA** on a **Diet Weight-Loss** dataset ($N = 78$ individuals), with full assumption validation (Shapiro-Wilk, Levene) and post-hoc pairwise comparisons (Tukey HSD).

> A full written report (in French) is available in [`Analyse_Des_Données - Rapport - Halepidis_Lion.pdf`](Analyse_Des_Donn%C3%A9es%20-%20Rapport%20-%20Halepidis_Lion.pdf).

---

## 1. PCA — OECD Better Life Index 2024

### Methodology
- **Preprocessing**: Mean imputation for missing values (notably `Time devoted to leisure and personal care` with 16/38 missing values) followed by z-score standardization (`StandardScaler`) to eliminate scale dominance across monetary and rate indicators.
- **Component selection**: Scree plot elbow retains **3 principal components** explaining **61.7%** of total variance (vs. 7 axes / 81.0% under Kaiser's $\lambda > 1$ rule).
- **Sensitivity analysis**: Re-running the full standardization and PCA without the two highest-leverage outliers (**Mexico** and **Colombia**) to verify structural robustness.

### Key Findings
| Component | Eigenvalue ($\lambda$) | Variance Explained | Cumulative Variance | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **PC1** | 9.335 | 37.3% | 37.3% | Overall development & quality-of-life gradient (`Life satisfaction` $r=0.85$, `GDP` $r=0.80$ vs. `Air pollution` $r=-0.74$) |
| **PC2** | 3.292 | 13.2% | 50.5% | Heterogeneous axis mixing household wealth ($r=0.70$) with long working hours ($r=0.64$) & homicide rate ($r=0.56$) |
| **PC3** | 2.791 | 11.2% | 61.7% | Labor market stress (`Long-term unemployment` $r=0.74$, `Labour market insecurity` $r=0.64$), driven by **Greece** (25.2%) & **Spain** (12.4%) |

- **Outlier impact**: Mexico and Colombia (5.3% of countries) contribute **31.0%** to PC1 and **24.6%** to PC2. Removing them reduces PC1 variance from **37.3% to 33.1%** while preserving its structure, proving **PC1 is robust** whereas **PC2 is heavily outlier-driven**.

### Key Visualizations
| Correlation Circle (PC1–PC2) | Country Projection (PC1–PC2) |
| :---: | :---: |
| ![Correlation Circle](Projet%20(ACP)%20BLI/resultats/04_cercle_correlations.png) | ![Country Projection](Projet%20(ACP)%20BLI/resultats/05_projection_individus.png) |

**Sensitivity Analysis (excluding Mexico & Colombia):**
![Sensitivity Analysis](Projet%20(ACP)%20BLI/resultats/06_comparaison_sans_outliers.png)

---

## 2. ANOVA — Diet Weight-Loss Efficacy

### Methodology
- **Target variable**: `weight_loss = preweight - weight6weeks` (in kg, positive = weight loss) across 3 diets (`A`: $n=24$, `B`: $n=27$, `C`: $n=27$) and gender (`F`: $43$, `M`: $35$).
- **Assumption checks** ($\alpha = 0.05$):
  - **Normality of residuals (Shapiro-Wilk)**: 1-way $W = 0.9921$ ($p = 0.9161$); 2-way $W = 0.9798$ ($p = 0.2527$) — **verified**.
  - **Homoscedasticity (Levene)**: median-centered $F = 0.6257$ ($p = 0.5377$); mean-centered $F = 0.6589$ ($p = 0.5204$) — **verified**.
- **Models**: One-Way ANOVA + Tukey HSD post-hoc test, followed by **Type II Two-Way ANOVA** (appropriate for unbalanced cell counts).

### Key Findings
- **One-Way ANOVA (`weight_loss ~ diet`)**: Significant diet effect — **$F(2, 75) = 6.20$, $p = 0.0032$**.
  - Mean weight loss: **Diet A** = `3.30 ± 2.24 kg`, **Diet B** = `3.03 ± 2.52 kg`, **Diet C** = `5.15 ± 2.40 kg`.
- **Tukey HSD Post-Hoc**:
  - **Diet C vs. Diet A**: `+1.85 kg` ($p = 0.0201$) — **significant**
  - **Diet C vs. Diet B**: `+2.12 kg` ($p = 0.0048$) — **significant**
  - **Diet B vs. Diet A**: `-0.27 kg` ($p = 0.9125$) — not significant
- **Two-Way ANOVA (`weight_loss ~ diet * gender`, Type II)**:
  - **Diet**: significant ($F(2, 72) = 6.31$, $p = 0.0030$)
  - **Gender**: not significant ($F(1, 72) = 0.024$, $p = 0.8771$; F: `3.89 kg` vs. M: `3.79 kg`)
  - **Diet × Gender interaction**: not significant ($F(2, 72) = 2.24$, $p = 0.1143$)

### Key Visualizations
| Weight-Loss Distribution by Diet & Gender | Diet × Gender Interaction Plot (95% CI) |
| :---: | :---: |
| ![Weight-Loss Distribution](Projet%20(ANOVA)%20Diet/resultats/02_distribution_perte_poids.png) | ![Interaction Plot](Projet%20(ANOVA)%20Diet/resultats/04_interaction_genre_regime.png) |

---

## Tech Stack
- **Data Manipulation**: `pandas`, `numpy`
- **Statistics & Machine Learning**: `scikit-learn`, `scipy`, `statsmodels`
- **Visualization**: `matplotlib`, `seaborn`

---

## Project Structure
```text
statistical-data-analysis-pca-anova/
├── README.md
├── requirements.txt
├── Analyse_Des_Données - Rapport - Halepidis_Lion.pdf
├── Projet (ACP) BLI/
│   ├── analyse_acp.py
│   ├── better-life-index-2024.csv
│   └── resultats/
└── Projet (ANOVA) Diet/
    ├── analyse_anova.py
    ├── diet.csv
    └── resultats/
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/hleo1304/statistical-data-analysis-pca-anova.git
cd statistical-data-analysis-pca-anova

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analyses (regenerates all figures in each resultats/ folder)
python "Projet (ACP) BLI/analyse_acp.py"
python "Projet (ANOVA) Diet/analyse_anova.py"
```

---

## Authors
- **Léo Halepidis** & **Maxence Lion**
