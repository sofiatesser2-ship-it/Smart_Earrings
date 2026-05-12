import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carica il dataset (Nome file aggiornato)
file_path = 'features_extraction.csv'
df = pd.read_csv(file_path)

# Definiamo la lista completa delle feature (incluse le nuove)
features = ['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2', 'LF_HF']

# Pulizia: rimuoviamo righe con valori mancanti per le feature selezionate
df = df.dropna(subset=features)

# --- 2. SEZIONE STATISTICA ---
print("==========================================")
print("--- ANALISI STATISTICA DEL DATASET ---")
print("==========================================\n")

print("1. Numero di campioni per ogni classe:")
print(df['Label'].value_counts())
print("\n" + "-"*30 + "\n")

print("2. Medie delle features per ogni classe (Valori Normalizzati):")
print(df.groupby('Label')[features].mean())
print("\n" + "="*42 + "\n")


# --- 3. GRAFICI ---
sns.set_theme(style="whitegrid")

# Creiamo una griglia 2x4 (per 7 feature, l'ultimo slot rimarrà vuoto o possiamo nasconderlo)
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
fig.suptitle('Confronto delle Feature HRV (Densità e Media)', fontsize=22, fontweight='bold')

# Appiattiamo l'array axes per ciclarlo facilmente
axes_flat = axes.flatten()

for i, feature in enumerate(features):
    ax = axes_flat[i]
    
    # Violin Plot (distribuzione)
    sns.violinplot(x='Label', y=feature, data=df, ax=ax, 
                   hue='Label', palette="viridis", legend=False, inner=None, alpha=0.7)
    
    # Point Plot (media con pallino nero)
    sns.pointplot(x='Label', y=feature, data=df, ax=ax, 
                  color='black', markers='o', linestyles='', 
                  errorbar=None, markersize=8)
    
    # Linea di riferimento Baseline (essendo dati normalizzati, la baseline è a 1.0)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    
    ax.set_title(f"{feature}", fontsize=16, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Valore Normalizzato', fontsize=12)

# Nascondiamo l'ottavo subplot (vuoto)
if len(features) < len(axes_flat):
    axes_flat[-1].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()