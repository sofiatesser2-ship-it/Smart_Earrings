import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# --- 1. FUNZIONI DI UTILITÀ ---
# ==========================================
def calculate_icc(g, c):
    """Calcola l'Intraclass Correlation Coefficient (ICC)."""
    data = np.stack([g, c], axis=1)
    n, k = data.shape
    ms_between = np.var(data.mean(axis=1), ddof=1) * k
    ms_within = data.var(axis=1, ddof=1).mean()
    ms_raters = np.var(data.mean(axis=0), ddof=1) * n
    ms_error = ((n-1)*(k-1)*ms_within - (k-1)*(ms_raters-ms_within)) / ((n-1)*(k-1))
    return (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + (k / n) * (ms_raters - ms_error))

# ==========================================
# --- 2. CONFIGURAZIONE PERCORSI ---
# ==========================================
# Trova la cartella corrente (models) e sale di un livello per la cartella principale
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
cartella_principale = os.path.dirname(cartella_corrente)

# Percorso del file Garmin (dentro 'models')
file_path_garmin = os.path.join(cartella_corrente, 'garmin140526.csv')

# Cerca tutti i file numerati nella cartella principale (es. 140526.1.csv)
# Usiamo glob per trovare tutti i CSV nella cartella principale
pattern_ricerca = os.path.join(cartella_principale, '140526.*.csv')
file_custom_list = glob.glob(pattern_ricerca)

if not file_custom_list:
    print(f"ATTENZIONE: Nessun file trovato con il pattern {pattern_ricerca}")
    print("Assicurati che i file (es. 140526.1.csv) siano nella cartella principale del progetto.")
    exit()

# Ordina i file per numero (opzionale, ma utile per l'ordine di stampa)
try:
    file_custom_list.sort(key=lambda f: int(os.path.basename(f).split('.')[1]))
except:
    file_custom_list.sort() # Fallback a ordinamento alfabetico se l'estrazione del numero fallisce

print(f"Trovati {len(file_custom_list)} file da analizzare.")

# ==========================================
# --- 3. CARICAMENTO DATI GARMIN ---
# ==========================================
# Lo facciamo una volta sola fuori dal ciclo
try:
    df_garmin = pd.read_csv(file_path_garmin, sep=';')
    df_garmin.columns = df_garmin.columns.str.strip()
    
    garmin_bpm = df_garmin['bpm_medio'].tolist()
    garmin_sdnn = df_garmin['sdnn'].tolist()
    garmin_rmssd = df_garmin['rmssd'].tolist()

except Exception as e:
    print(f"ERRORE CRITICO: Impossibile caricare il file Garmin: {e}")
    exit()


# ==========================================
# --- 4. CICLO DI ANALISI (BATCH PROCESSING) ---
# ==========================================
for file_corrente in file_custom_list:
    nome_file = os.path.basename(file_corrente)
    print(f"\n\n{'='*90}")
    print(f"--- INIZIO ANALISI: {nome_file} ---")
    print(f"{'='*90}")

    # --- 4.1 Caricamento Dati Custom Correnti ---
    try:
        df_custom = pd.read_csv(file_corrente, sep=';')
        if len(df_custom.columns) < 2: 
            df_custom = pd.read_csv(file_corrente, sep=',')
        df_custom.columns = df_custom.columns.str.strip()
        
        custom_bpm = df_custom['BPM'].tolist()
        custom_sdnn = df_custom['SDNN'].tolist()
        custom_rmssd = df_custom['RMSSD'].tolist()
    except Exception as e:
        print(f"Errore nella lettura di {nome_file}: {e}. Salto al prossimo file.")
        continue # Passa al prossimo file nel ciclo

    # --- 4.2 Allineamento Dati ---
    min_len = min(len(custom_bpm), len(garmin_bpm))
    if min_len == 0:
        print(f"Attenzione: Dati insufficienti in {nome_file}. Salto al prossimo.")
        continue

    data_input = {
        'BPM_Medio': {'custom': custom_bpm[:min_len], 'garmin': garmin_bpm[:min_len]},
        'SDNN':      {'custom': custom_sdnn[:min_len], 'garmin': garmin_sdnn[:min_len]},
        'RMSSD':     {'custom': custom_rmssd[:min_len], 'garmin': garmin_rmssd[:min_len]}
    }

    summary_list = []

    # --- 4.3 Generazione Grafici ---
    # Creiamo una figura unica con 2 righe (Scatter sopra, Bland-Altman sotto)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Analisi Completa: {nome_file} vs Garmin', fontsize=18, fontweight='bold', y=0.98)

    for i, (m_name, vals) in enumerate(data_input.items()):
        c, g = np.array(vals['custom']), np.array(vals['garmin'])
        
        # --- Calcolo Metriche ---
        icc = calculate_icc(g, c)
        r2 = r2_score(g, c)
        mae, rmse = mean_absolute_error(g, c), np.sqrt(mean_squared_error(g, c))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((g - c) / g)) * 100
            mape = np.nan_to_num(mape, posinf=0, neginf=0) 
        _, p_val = stats.wilcoxon(g, c)
        summary_list.append([m_name, mae, rmse, mape, r2, icc, p_val])

        # --- Grafico 1: Correlazione (Scatter) - Riga 0 ---
        ax_scatter = axes[0, i]
        center = (g.min() + g.max()) / 2
        span = (g.max() - g.min()) * 1.5 
        ax_scatter.set_xlim(center - span, center + span)
        ax_scatter.set_ylim(center - span, center + span)

        ax_scatter.scatter(g, c, color='dodgerblue', edgecolor='k', s=80, zorder=3)
        ax_scatter.plot([-200, 300], [-200, 300], 'r--', alpha=0.6, label='x=y')
        ax_scatter.set_title(f'Scatter: {m_name}\nICC: {icc:.3f}')
        ax_scatter.grid(alpha=0.2)

        # --- Grafico 2: Bland-Altman - Riga 1 ---
        ax_bland = axes[1, i]
        diffs, means = c - g, (c + g) / 2
        bias, sd_diff = np.mean(diffs), np.std(diffs, ddof=1)
        uLoA, lLoA = bias + 1.96 * sd_diff, bias - 1.96 * sd_diff
        
        y_limit = max(abs(uLoA), abs(lLoA)) * 2.5 
        if y_limit == 0: y_limit = 10 # Evita warning se tutti i valori sono uguali
        ax_bland.set_ylim(-y_limit, y_limit)
        
        ax_bland.scatter(means, diffs, color='mediumpurple', edgecolor='k', s=80, zorder=3)
        ax_bland.axhline(bias, color='red', lw=2, label=f'Bias: {bias:.2f}')
        ax_bland.axhline(uLoA, color='black', linestyle=':', alpha=0.6)
        ax_bland.axhline(lLoA, color='black', linestyle=':', alpha=0.6)
        ax_bland.axhline(0, color='gray', lw=1, alpha=0.3)
        ax_bland.set_title(f'Bland-Altman: {m_name}')
        ax_bland.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Lascia spazio per il titolo principale
    plt.show(block=True) # Mette in pausa l'esecuzione per farti vedere i grafici di OGNI file. Chiudi la finestra per passare al successivo.

    # --- 4.4 Stampa Tabella ---
    df_final = pd.DataFrame(summary_list, columns=['Metrica', 'MAE', 'RMSE', 'MAPE (%)', 'R2', 'ICC', 'p-value'])
    print(f"\nTABELLA RIASSUNTIVA ({nome_file})")
    print("-" * 85)
    print(df_final.to_string(index=False, justify='center'))
    print("-" * 85)

print("\n=== ELABORAZIONE BATCH COMPLETATA ===")