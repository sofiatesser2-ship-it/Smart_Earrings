import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carica il dataset
file_path = "features_extraction_new_dataset.csv"
df = pd.read_csv(file_path)

features = ["BPM", "RMSSD", "SDNN", "PNN50", "SD1", "SD2", "LF_HF"]
df = df.dropna(subset=features)

# --- 2. SEZIONE STATISTICA ---
print("==========================================")
print("--- ANALISI STATISTICA DEL DATASET ---")
print("==========================================\n")
print("1. Numero di campioni per ogni classe:")
print(df["Label"].value_counts())
print("\n" + "-" * 30 + "\n")
print("2. Medie delle features per ogni classe (Valori Normalizzati):")
print(df.groupby("Label")[features].mean())
print("\n" + "=" * 42 + "\n")


# --- 3. GRAFICI OTTIMIZZATI ---
# Impostiamo un tema moderno e minimalista
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
    }
)

# Palette personalizzata: un blu/verde rilassante per Baseline, un rosso/arancio per Stress
colors = {"Baseline": "#3a86ff", "Stress": "#ff006e"}

# Griglia 2x4 (più compatta ed elegante: 20x10)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(
    "Analisi Comparativa delle Feature HRV: Baseline vs Stress",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

axes_flat = axes.flatten()

for i, feature in enumerate(features):
    ax = axes_flat[i]

    # Aggiungiamo un leggero sfondo colorato per far risaltare i singoli subplot
    ax.set_facecolor("#fcfcfc")

    # A) VIOLIN PLOT: Solo contorno tagliato sui dati reali (cut=0) per evitare code artificiali
    sns.violinplot(
        x="Label",
        y=feature,
        data=df,
        ax=ax,
        hue="Label",
        palette=colors,
        legend=False,
        inner=None,
        alpha=0.4,
        cut=0,
    )

    # B) STRIP PLOT: Mostra i singoli record come punti semitrasparenti (jitter controllato)
    # Questo permette di vedere la reale densità dei dati senza appesantire
    sns.stripplot(
        x="Label",
        y=feature,
        data=df,
        ax=ax,
        hue="Label",
        palette=colors,
        legend=False,
        size=4,
        alpha=0.4,
        jitter=0.2,
        dodge=False,
    )

    # C) BOXPLOT MINIATURIZZATO: Sostituisce il point plot.
    # Mostra mediana, quartili e media (con la "showmeans") in modo nativo e perfettamente allineato
    sns.boxplot(
        x="Label",
        y=feature,
        data=df,
        ax=ax,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "gray", "alpha": 0.5},
        capprops={"color": "gray", "alpha": 0.5},
        width=0.15,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
        showfliers=False,  # Gli outlier si vedono già dallo stripplot
    )

    # D) LINEA DI RIFERIMENTO (Baseline a 1.0)
    ax.axhline(1.0, color="#2b2d42", linestyle=":", alpha=0.5, linewidth=1.5)

    # Pulizia etichette e titoli
    ax.set_title(feature, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Valore Normalizzato" if i in [0, 4] else "")

    # Rimuove il bordo superiore e destro per un look pulito
    sns.despine(ax=ax, left=False, bottom=False)

# Nascondiamo l'ottavo subplot vuoto
if len(features) < len(axes_flat):
    axes_flat[-1].set_visible(False)

# Gestione spazio ottimale per evitare sovrapposizioni del titolo principale
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)
plt.show()