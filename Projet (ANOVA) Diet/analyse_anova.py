# Analyse de la Variance (ANOVA) - Jeu de données Diet

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "diet.csv")
OUT_DIR = os.path.join(BASE_DIR, "resultats")
os.makedirs(OUT_DIR, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
sns.set_theme(style="whitegrid")

# 1. Préparation des données
print("\n--- 1. Préparation des données ---")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print(f"Dimensions : {df.shape[0]} individus x {df.shape[1]} variables")
print("\nValeurs manquantes :")
print(df.isna().sum())

df["gender"] = df["gender"].astype(str).str.strip()
df["diet"] = df["diet"].astype(str).str.strip()

# Calcul de la perte de poids (positive = perte de poids)
df["weight_loss"] = df["preweight"] - df["weight6weeks"]

print("\nEffectifs par régime et par genre :")
print(pd.crosstab(df["diet"], df["gender"], margins=True))

print("\nStatistiques descriptives par régime :")
print(df.groupby("diet")["weight_loss"].describe())

print("\nStatistiques descriptives par régime et genre :")
print(df.groupby(["diet", "gender"])["weight_loss"].describe())

# 2. Exploration visuelle
print("\n--- 2. Génération des graphiques exploratoires ---")

# Nuage de points : poids final en fonction du poids initial
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="preweight", y="weight6weeks", hue="diet", style="gender",
                s=65, ax=ax)
lim_min = min(df["preweight"].min(), df["weight6weeks"].min()) - 2
lim_max = max(df["preweight"].max(), df["weight6weeks"].max()) + 2
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="Droite y = x")
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("Poids initial (preweight, kg)")
ax.set_ylabel("Poids après 6 semaines (weight6weeks, kg)")
ax.set_title("Poids final en fonction du poids initial")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_scatter_poids.png"), dpi=300)
plt.close()

# Distribution de la perte de poids (violon + boxplot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.violinplot(data=df, x="diet", y="weight_loss", hue="gender", inner=None, alpha=0.4, ax=ax1)
sns.boxplot(data=df, x="diet", y="weight_loss", hue="gender", width=0.35, ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], title="Genre")
ax1.set_title("Perte de poids par régime et genre")
ax1.set_xlabel("Régime")
ax1.set_ylabel("Perte de poids (kg)")

sns.boxplot(data=df, x="diet", y="weight_loss", width=0.45, ax=ax2)
sns.stripplot(data=df, x="diet", y="weight_loss", hue="gender", size=6, ax=ax2)
ax2.set_title("Perte de poids par régime (points individuels)")
ax2.set_xlabel("Régime")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_distribution_perte_poids.png"), dpi=300)
plt.close()

# 3. Vérification des hypothèses préalables
print("\n--- 3. Vérification des hypothèses (Normalité et Homoscédasticité) ---")

model1 = smf.ols("weight_loss ~ C(diet)", data=df).fit()
model2 = smf.ols("weight_loss ~ C(diet) * C(gender)", data=df).fit()

# Test de Shapiro-Wilk sur les résidus
w1, p_val1 = stats.shapiro(model1.resid)
print(f"\nShapiro-Wilk (modèle 1 facteur) : W = {w1:.4f}, p-value = {p_val1:.4f}")
if p_val1 > 0.05:
    print("-> p > 0.05 : On ne rejette pas H0, les résidus suivent une loi normale.")
else:
    print("-> p <= 0.05 : On rejette H0 (normalité non respectée).")

w2, p_val2 = stats.shapiro(model2.resid)
print(f"Shapiro-Wilk (modèle 2 facteurs) : W = {w2:.4f}, p-value = {p_val2:.4f}")

# Q-Q Plot des résidus
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sm.qqplot(model1.resid, line="s", ax=ax1)
ax1.set_title("Q-Q Plot des résidus (ANOVA 1 facteur)")
sm.qqplot(model2.resid, line="s", ax=ax2)
ax2.set_title("Q-Q Plot des résidus (ANOVA 2 facteurs)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_qqplot_residus.png"), dpi=300)
plt.close()

# Test de Levene (homogénéité des variances)
gA = df[df["diet"] == "A"]["weight_loss"]
gB = df[df["diet"] == "B"]["weight_loss"]
gC = df[df["diet"] == "C"]["weight_loss"]

stat_lev1, p_lev1 = stats.levene(gA, gB, gC, center="median")
stat_lev2, p_lev2 = stats.levene(gA, gB, gC, center="mean")
print(f"\nTest de Levene (médiane) : F = {stat_lev1:.4f}, p-value = {p_lev1:.4f}")
print(f"Test de Levene (moyenne) : F = {stat_lev2:.4f}, p-value = {p_lev2:.4f}")
if p_lev1 > 0.05:
    print("-> p > 0.05 : On ne rejette pas H0, il y a homoscédasticité entre les régimes.")

# 4. ANOVA à 1 facteur (Régime) et test post-hoc de Tukey
print("\n--- 4. ANOVA à 1 facteur (Régime) ---")

anova1 = anova_lm(model1)
anova1["mean_sq"] = anova1["sum_sq"] / anova1["df"]
anova1 = anova1[["sum_sq", "df", "mean_sq", "F", "PR(>F)"]]
print(anova1)

p_anova1 = anova1.loc["C(diet)", "PR(>F)"]
if p_anova1 < 0.05:
    print(f"\n-> p = {p_anova1:.4f} < 0.05 : On rejette H0, le régime a un impact significatif.")
    print("\nTest post-hoc de Tukey HSD :")
    tukey = pairwise_tukeyhsd(endog=df["weight_loss"], groups=df["diet"], alpha=0.05)
    print(tukey)

# 5. ANOVA à 2 facteurs (Régime, Genre et Interaction)
print("\n--- 5. ANOVA à 2 facteurs (Régime * Genre) ---")

# Sommes des carrés de type II car le plan n'est pas équilibré
anova2 = anova_lm(model2, typ=2)
anova2["mean_sq"] = anova2["sum_sq"] / anova2["df"]
anova2 = anova2[["sum_sq", "df", "mean_sq", "F", "PR(>F)"]]
print(anova2)

# Graphique d'interaction
plt.figure(figsize=(8, 6))
sns.pointplot(data=df, x="diet", y="weight_loss", hue="gender",
              dodge=0.1, markers=["o", "s"], capsize=0.1, errorbar=("ci", 95))
plt.xlabel("Régime")
plt.ylabel("Perte de poids moyenne (kg)")
plt.title("Graphique d'interaction : perte de poids selon le régime et le genre (IC 95 %)")
plt.legend(title="Genre")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_interaction_genre_regime.png"), dpi=300)
plt.close()

print("\nAnalyse ANOVA terminée.")
