# Analyse en Composantes Principales (ACP) - Better Life Index 2024

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "better-life-index-2024.csv")
OUT_DIR = os.path.join(BASE_DIR, "resultats")
os.makedirs(OUT_DIR, exist_ok=True)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

# 1. Pré-traitement et matrice de corrélation
print("\n--- 1. Pré-traitement et matrice de corrélation ---")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df.set_index("Country")

# On ne garde que les variables quantitatives
X = df.select_dtypes(include=[np.number]).copy()
print(f"Dimensions : {X.shape[0]} pays x {X.shape[1]} variables")

# Affichage et imputation des valeurs manquantes par la moyenne
na = X.isna().sum()
print("\nValeurs manquantes par variable :")
print(na[na > 0])
X = X.fillna(X.mean())

corr = X.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True,
            fmt=".2f", annot_kws={"size": 6}, square=True, linewidths=0.3)
plt.title("Matrice de corrélation - Better Life Index 2024")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_matrice_correlation.png"), dpi=300)
plt.close()

# 2. Standardisation (centrage-réduction)
print("\n--- 2. Standardisation (centrage-réduction) ---")

# Centrage-réduction nécessaire car les variables ont des unités très différentes
scaler = StandardScaler()
Z_mat = scaler.fit_transform(X)
Z = pd.DataFrame(Z_mat, index=X.index, columns=X.columns)
n, p = Z.shape

plt.figure(figsize=(15, 7))
sns.boxplot(data=Z, color="steelblue", fliersize=3)
plt.axhline(0, color="red", lw=0.8, ls="--")
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Valeur standardisée (z-score)")
plt.title("Boxplots des variables après centrage-réduction")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_boxplot_standardise.png"), dpi=300)
plt.close()

# 3. Calcul de l'ACP
print("\n--- 3. Calcul de l'ACP ---")

pca = PCA()
F = pca.fit_transform(Z_mat)
eigvec = pca.components_.T.copy()
eigval = pca.explained_variance_ratio_ * p

# Orientation positive des vecteurs propres pour faciliter la lecture
for j in range(p):
    if eigvec[:, j].sum() < 0:
        eigvec[:, j] *= -1
        F[:, j] *= -1

axes = [f"Axe {i + 1}" for i in range(p)]
var_pct = pca.explained_variance_ratio_ * 100
cum_pct = np.cumsum(var_pct)

tab_vp = pd.DataFrame({
    "Valeur propre": eigval,
    "Variance expliquée (%)": var_pct,
    "Variance cumulée (%)": cum_pct
}, index=axes)
print(tab_vp.head(10))
print(f"\nNombre d'axes avec valeur propre > 1 (Kaiser) : {int((eigval > 1).sum())}")

# Éboulis des valeurs propres et variance cumulée
fig, ax1 = plt.subplots(figsize=(10, 6))
x_axes = np.arange(1, p + 1)
ax1.bar(x_axes, eigval, color="steelblue", alpha=0.85, label="Valeur propre")
ax1.plot(x_axes, eigval, "o-", color="navy")
ax1.axhline(1, color="red", ls="--", lw=1, label="Seuil de Kaiser (λ = 1)")
ax1.set_xlabel("Axe factoriel")
ax1.set_ylabel("Valeur propre")
ax1.set_xticks(x_axes)

ax2 = ax1.twinx()
ax2.plot(x_axes, cum_pct, "s-", color="darkorange", label="Variance cumulée (%)")
ax2.axhline(70, color="darkorange", ls=":", lw=1)
ax2.set_ylabel("Variance cumulée (%)")
ax2.set_ylim(0, 105)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right")
plt.title("Éboulis des valeurs propres et variance cumulée")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_eboulis_variance_cumulee.png"), dpi=300)
plt.close()

# 4. Métriques d'interprétation (Axes 1, 2 et 3)
print("\n--- 4. Métriques d'interprétation (Axes 1 à 3) ---")

cols = ["Axe 1", "Axe 2", "Axe 3"]
G = eigvec * np.sqrt(eigval)

coord_ind = pd.DataFrame(F[:, :3], index=Z.index, columns=cols)
coord_var = pd.DataFrame(G[:, :3], index=Z.columns, columns=cols)

# Qualité de représentation (cos²)
cos2_var = pd.DataFrame(G[:, :3] ** 2, index=Z.columns, columns=cols)
cos2_var["Somme (axes 1-3)"] = cos2_var.sum(axis=1)

d2 = (Z_mat ** 2).sum(axis=1)
cos2_ind = pd.DataFrame(F[:, :3] ** 2 / d2.reshape(-1, 1), index=Z.index, columns=cols)
cos2_ind["Somme (axes 1-3)"] = cos2_ind.sum(axis=1)

# Contributions (%)
ctr_var = pd.DataFrame((eigvec[:, :3] ** 2) * 100, index=Z.columns, columns=cols)
ctr_ind = pd.DataFrame((F[:, :3] ** 2) / (n * eigval[:3]) * 100, index=Z.index, columns=cols)

print("\nCoordonnées des variables :")
print(coord_var)
print("\ncos² des variables :")
print(cos2_var)
print("\nContributions des variables (%) :")
print(ctr_var)

print("\nCoordonnées des individus :")
print(coord_ind)
print("\ncos² des individus :")
print(cos2_ind)
print("\nContributions des individus (%) :")
print(ctr_ind)


def tracer_cercle(ax, cv, cos2v, a, b, vp, titre_fig):
    ca, cb = f"Axe {a}", f"Axe {b}"
    cercle = plt.Circle((0, 0), 1, fill=False, color="black", lw=1.2)
    ax.add_patch(cercle)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)

    qual = cos2v[ca] + cos2v[cb]
    sc = ax.scatter(cv[ca], cv[cb], c=qual, cmap="viridis", vmin=0, vmax=1, s=15)

    for nom in cv.index:
        x0, y0 = cv.loc[nom, ca], cv.loc[nom, cb]
        ax.arrow(0, 0, x0, y0, head_width=0.02, length_includes_head=True,
                 color=plt.cm.viridis(qual[nom]), alpha=0.85)
        ax.annotate(nom, (x0, y0), fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{ca} ({vp[a - 1]:.1f} %)")
    ax.set_ylabel(f"{cb} ({vp[b - 1]:.1f} %)")
    ax.set_title(titre_fig)
    return sc


def tracer_individus(ax, ci, cos2i, a, b, vp, titre_fig):
    ca, cb = f"Axe {a}", f"Axe {b}"
    qual = cos2i[ca] + cos2i[cb]
    sc = ax.scatter(ci[ca], ci[cb], c=qual, cmap="viridis", vmin=0, vmax=1,
                    s=45, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)

    for pays in ci.index:
        x0, y0 = ci.loc[pays, ca], ci.loc[pays, cb]
        dy = -9 if pays in ["Luxembourg", "Colombia", "Chile"] else 4
        ax.annotate(pays, (x0, y0), fontsize=7.5, xytext=(4, dy), textcoords="offset points")

    ax.margins(0.12)
    ax.set_xlabel(f"{ca} ({vp[a - 1]:.1f} %)")
    ax.set_ylabel(f"{cb} ({vp[b - 1]:.1f} %)")
    ax.set_title(titre_fig)
    return sc


# 5. Projections sur le plan 1-2
print("\n--- 5. Projections sur le plan factoriel 1-2 ---")

fig, ax = plt.subplots(figsize=(10, 9))
sc = tracer_cercle(ax, coord_var, cos2_var, 1, 2, var_pct, "Cercle des corrélations (axes 1 et 2)")
fig.colorbar(sc, ax=ax, shrink=0.7, label="cos² sur le plan 1-2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_cercle_correlations.png"), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(12, 9))
sc = tracer_individus(ax, coord_ind, cos2_ind, 1, 2, var_pct,
                      "Projection des individus sur le plan factoriel (axes 1 et 2)")
fig.colorbar(sc, ax=ax, label="cos² sur le plan 1-2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_projection_individus.png"), dpi=300)
plt.close()

# 6. Analyse de sensibilité (sans Mexico et Colombia)
print("\n--- 6. Analyse de sensibilité (sans Mexico et Colombia) ---")

X_sans = df.select_dtypes(include=[np.number]).drop(index=["Mexico", "Colombia"])
X_sans = X_sans.fillna(X_sans.mean())
Z_sans = StandardScaler().fit_transform(X_sans)

pca_s = PCA()
F_s = pca_s.fit_transform(Z_sans)
eigvec_s = pca_s.components_.T.copy()
eigval_s = pca_s.explained_variance_ratio_ * p
vp_s = pca_s.explained_variance_ratio_ * 100

for j in range(p):
    if eigvec_s[:, j].sum() < 0:
        eigvec_s[:, j] *= -1
        F_s[:, j] *= -1

G_s = eigvec_s * np.sqrt(eigval_s)
coord_ind_s = pd.DataFrame(F_s[:, :3], index=X_sans.index, columns=cols)
coord_var_s = pd.DataFrame(G_s[:, :3], index=X_sans.columns, columns=cols)
cos2_var_s = pd.DataFrame(G_s[:, :3] ** 2, index=X_sans.columns, columns=cols)
d2_s = (Z_sans ** 2).sum(axis=1)
cos2_ind_s = pd.DataFrame(F_s[:, :3] ** 2 / d2_s.reshape(-1, 1), index=X_sans.index, columns=cols)

comp = pd.DataFrame({
    "Var (%) complet": var_pct[:5],
    "Cumul (%) complet": cum_pct[:5],
    "Var (%) sans outliers": vp_s[:5],
    "Cumul (%) sans outliers": np.cumsum(vp_s)[:5]
}, index=axes[:5])
print(comp)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
sc1 = tracer_individus(ax1, coord_ind_s, cos2_ind_s, 1, 2, vp_s,
                       "Individus (axes 1-2) sans Mexico et Colombia")
sc2 = tracer_cercle(ax2, coord_var_s, cos2_var_s, 1, 2, vp_s,
                    "Cercle des corrélations (axes 1-2) sans Mexico et Colombia")
fig.colorbar(sc1, ax=ax1, shrink=0.7, label="cos² sur le plan 1-2")
fig.colorbar(sc2, ax=ax2, shrink=0.7, label="cos² sur le plan 1-2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_comparaison_sans_outliers.png"), dpi=300)
plt.close()

# 7. Plan factoriel 1-3
print("\n--- 7. Projection sur le plan factoriel 1-3 ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
sc1 = tracer_cercle(ax1, coord_var, cos2_var, 1, 3, var_pct, "Cercle des corrélations (axes 1 et 3)")
sc2 = tracer_individus(ax2, coord_ind, cos2_ind, 1, 3, var_pct, "Projection des individus (axes 1 et 3)")
fig.colorbar(sc1, ax=ax1, shrink=0.7, label="cos² sur le plan 1-3")
fig.colorbar(sc2, ax=ax2, shrink=0.7, label="cos² sur le plan 1-3")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_projection_plan_1_3.png"), dpi=300)
plt.close()

print("\nAnalyse ACP terminée.")
