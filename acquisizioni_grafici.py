import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Impostazione dello stile dei grafici
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

def plot_hrv_comparison(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Errore: Il file '{csv_path}' non esiste. Esegui prima l'estrazione delle feature!")
        return

    # 1. Carica il dataset
    df = pd.read_csv(csv_path)
    print(f"📊 Dati caricati correttamente. Record totali: {len(df)}")
    print(df.groupby(['Subject', 'Label']).size().unstack(fill_value=0))

    # Elenco delle metriche principali da visualizzare
    metrics = ['BPM', 'RMSSD', 'SDNN']
    
    # --- GRAFICO 1: BOXPLOT COMPARATIVI ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    fig.suptitle('Confronto delle Metriche HRV: Baseline vs Stress (Dati Normalizzati)', fontweight='bold', y=1.02)

    colors = {'Baseline': '#4C72B0', 'Stress': '#C44E52'}

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            # Creazione del Boxplot + Stripplot (mostra i singoli punti per vedere la densità)
            sns.boxplot(ax=axes[i], x='Label', y=metric, data=df, palette=colors, width=0.4, showfliers=False)
            sns.stripplot(ax=axes[i], x='Label', y=metric, data=df, color='black', alpha=0.3, size=4, jitter=0.15)
            
            axes[i].set_title(f'Distribuzione {metric}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Valore Normalizzato (Rapporto)')
            
            # Linea di riferimento a 1.0 (essendo i dati rapportati alla media della baseline)
            axes[i].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # --- GRAFICO 2: ANDAMENTO TEMPORALE PER SOGGETTO ---
    # Scegliamo una metrica rappresentativa come il BPM o l'RMSSD per vedere come varia nel tempo
    metric_trend = 'BPM'
    subjects = df['Subject'].unique()
    
    if metric_trend in df.columns and len(subjects) > 0:
        plt.figure(figsize=(14, 6))
        
        for sub in subjects:
            sub_df = df[df['Subject'] == sub].copy()
            # Creiamo un indice temporale fittizio basato sull'ordine delle finestre estratte
            sub_df['Window_Index'] = range(len(sub_df))
            
            # Disegna la linea continua per il soggetto
            plt.plot(sub_df['Window_Index'], sub_df[metric_trend], label=f'{sub.upper()}', alpha=0.7, marker='o', markersize=3)
            
            # Evidenzia graficamente dove si passa da Baseline a Stress nel tempo
            # Troviamo il punto di transizione
            stress_indices = sub_df[sub_df['Label'] == 'Stress']['Window_Index']
            if not stress_indices.empty:
                min_stress = stress_indices.min()
                plt.axvspan(min_stress, sub_df['Window_Index'].max(), color='#C44E52', alpha=0.03)

        plt.title(f'Andamento Temporale del {metric_trend} nelle Finestre Mobili', fontweight='bold')
        plt.xlabel('Indice Finestra Temporale (Avanzamento nel tempo)')
        plt.ylabel(f'{metric_trend} Normalizzato')
        
        # Linea di base
        plt.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='Livello Baseline Medio')
        
        # Legenda pulita senza duplicati per le zone colorate
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Rileva in automatico la cartella del progetto GitHub
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    # Se questo script si trova nella cartella 'models/', sale di un livello alla radice del progetto
    if os.path.basename(BASE_PATH) == 'models':
        BASE_PATH = os.path.dirname(BASE_PATH)
        
    CSV_FILE = os.path.join(BASE_PATH, 'features_extraction_new_dataset.csv')
    
    plot_hrv_comparison(CSV_FILE)