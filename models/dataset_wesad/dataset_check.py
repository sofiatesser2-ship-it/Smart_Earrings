import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Carica il dataset
file_path = "features_extraction.csv"
df = pd.read_csv(file_path)

features = ["BPM", "RMSSD", "SDNN", "PNN50", "SD1", "SD2", "LF_HF"]
df = df.dropna(subset=features)

# --- 2. CONFIGURAZIONE STILE ---
sns.set_theme(style="white", context="paper")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
    }
)

# Definizione esplicita dell'ordine delle categorie sull'asse X
target_order = ["Baseline", "Social_Stress", "Cognitive_Stress"]

# Palette colori aggiornata a 3 categorie (Colori pieni e moderni)
colors = {
    "Baseline": "#3a86ff",  # Blu
    "Social_Stress": "#ff006e",  # Magenta/Rosa forte
    "Cognitive_Stress": "#ffbe0b",  # Giallo ambra/Arancio
}

fig, axes = plt.subplots(2, 4, figsize=(22, 10))  # Allargato leggermente per fare spazio a 3 categorie
fig.suptitle(
    "Features HRV: Baseline vs Stress Conditions",
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
        order=target_order,  # Forza l'ordine delle colonne
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

    # B) VIOLIN PLOT
    sns.violinplot(
        x="Label",
        y=feature,
        data=df,
        order=target_order,  # Forza l'ordine delle colonne
        ax=ax,
        hue="Label",
        palette=colors,
        legend=False,
        inner=None,
        alpha=0.6,
        cut=0,
        linewidth=1.2,
        zorder=2,
    )

    # C) INDICATORI STATISTICI PULITI (Ciclo aggiornato a 3 categorie)
    for group_idx, label in enumerate(target_order):
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

    # D) LINEA DI RIFERIMENTO BASELINE (Tratteggiata a 1.0)
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