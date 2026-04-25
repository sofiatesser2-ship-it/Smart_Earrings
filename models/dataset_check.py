import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurazione stile
plt.style.use('seaborn-v0_8-muted')
sns.set_palette("husl")

def plot_comparison(file_path):
    if not os.path.exists(file_path):
        print("File non trovato!")
        return

    df = pd.read_csv(file_path)
    
    # Definiamo l'ordine delle classi per il grafico
    order = ['Baseline', 'Social_Stress', 'Cognitive_Stress']
    features = ['BPM', 'RMSSD', 'SDNN', 'LF_HF']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Confronto Fisiologico: Baseline vs Social Stress vs Cognitive Stress', fontsize=20)

    for i, feat in enumerate(features):
        ax = axes[i//2, i%2]
        
        # Violin plot per vedere la densità dei dati
        sns.violinplot(x='Label', y=feat, data=df, order=order, ax=ax, inner="quartile")
        
        # Aggiungiamo un punto per la media per chiarezza
        sns.pointplot(x='Label', y=feat, data=df, order=order, ax=ax, color='black', markers='D', errorbar=None)
        
        ax.set_title(f'Andamento di {feat}', fontsize=14, fontweight='bold')
        ax.axhline(1.0, ls='--', color='red', alpha=0.3) # Linea di riferimento Baseline
        ax.set_ylabel('Rapporto rispetto alla Baseline')
        ax.set_xlabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('confronto_3_classi.png', dpi=300)
    
    # Calcolo differenze percentuali medie
    print("\n--- ANALISI DELLE DIFFERENZE (Medie Normalizzate) ---")
    summary = df.groupby('Label')[features].mean().reindex(order)
    print(summary)
    
    # Calcolo variazione percentuale rispetto alla Baseline
    diff = ((summary.loc[['Social_Stress', 'Cognitive_Stress']] - 1.0) * 100).round(2)
    print("\nVariazione % rispetto alla Baseline:")
    print(diff)

if __name__ == "__main__":
    import os
    plot_comparison('wesad_complete_ratio.csv')
    plt.show()