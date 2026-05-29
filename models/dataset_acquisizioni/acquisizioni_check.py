import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carica il dataset
file_path = "features_extraction_binary.csv"
df = pd.read_csv(file_path)

features = ["BPM", "RMSSD", "SDNN", "PNN50", "SD1", "SD2", "LF_HF"]
df = df.dropna(subset=features)

# --- 2. CONFIGURAZIONE STILE ---
# Usiamo 'white' per eliminare completamente le linee grigie di sfondo (grid)
sns.set_theme(style="white", context="paper")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
    }
)

# Colori pieni e moderni
colors = {"Baseline": "#3a86ff", "Stress": "#ff006e"}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(
    " Features HRV: Baseline vs Stress",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

axes_flat = axes.flatten()

for i, feature in enumerate(features):
    ax = axes_flat[i]

    # Sfondo grigio chiarissimo neutro per ogni box
    ax.set_facecolor("#f8f9fa")

    # A) STRIP PLOT (I punti colorati sullo sfondo)
    sns.stripplot(
        x="Label",
        y=feature,
        data=df,
        ax=ax,
        hue="Label",
        palette=colors,
        legend=False,
        size=4,
        alpha=0.25,
        jitter=0.18,
        dodge=False,
        zorder=1,
    )

    # B) VIOLIN PLOT (Senza rettangoli o linee bianche interne: inner=None)
    sns.violinplot(
        x="Label",
        y=feature,
        data=df,
        ax=ax,
        hue="Label",
        palette=colors,
        legend=False,
        inner=None,  # Rimuove il rettangolo nero e la linea bianca di default!
        alpha=0.6,
        cut=0,
        linewidth=1.2,
        zorder=2,
    )

    # C) INDICATORI STATISTICI PULITI (Disegnati a mano per massimo controllo)
    # Calcoliamo media e mediana per ogni classe per metterle graficamente in modo elegante
    for group_idx, label in enumerate(["Baseline", "Stress"]):
        group_data = df[df["Label"] == label][feature]
        if not group_data.empty:
            mean_val = group_data.mean()
            median_val = group_data.median()

            # Disegna una linea nera sottile per la Mediana
            ax.hlines(
                y=median_val,
                xmin=group_idx - 0.15,
                xmax=group_idx + 0.15,
                colors="#212529",
                linewidth=2,
                zorder=3,
            )

            # Disegna un pallino nero pulito per la Media
            ax.scatter(
                x=group_idx,
                y=mean_val,
                color="black",
                s=35,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )

    # D) LINEA DI RIFERIMENTO BASELINE (Tratteggiata rossa/scura a 1.0)
    ax.axhline(1.0, color="#6c757d", linestyle="--", alpha=0.7, linewidth=1.2)

    # Pulizia etichette e grafiche
    ax.set_title(feature, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized Value" if i in [0, 4] else "")

    # Rimuove i bordi superflui per un design minimale
    sns.despine(ax=ax, left=False, bottom=False)

# Nascondiamo l'ottavo subplot vuoto in modo pulito
if len(features) < len(axes_flat):
    axes_flat[-1].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)
plt.show()