import heartpy as hp
import pandas as pd

# 1. Caricamento dati
path_file = 'ppg_200Hz_filtrato.csv'
df = pd.read_csv(path_file)

# Sostituisci 'signal' con il nome reale della tua colonna pre-filtrata
signal_data = df['Final_Result'].values

# 2. Parametri
fs = 200.0

try:
    # 3. Analisi Diretta
    # Usiamo direttamente hp.process senza passare per hp.filter_signal
    # Il parametro report_time=False serve solo a non stampare il tempo di calcolo
    wd, m = hp.process(signal_data, sample_rate=fs)

    # 4. Estrazione Feature (Stampa Formattata)
    bpm_medio = m['bpm']
    ibi_medio = m['ibi']
    sdnn = m['sdnn']
    rmssd = m['rmssd']

    # Battiti totali al netto di quelli eventualmente scartati da HeartPy
    # perché non conformi alla media (outlier residui)
    battiti_totali = len(wd['peaklist']) - len(wd['removed_beats'])

    print(f"BPM Medio:      {bpm_medio:.2f}")
    print(f"RR Medio (IBI): {ibi_medio:.2f} ms")
    print(f"SDNN:           {sdnn:.2f} ms")
    print(f"RMSSD:          {rmssd:.2f} ms")
    print(f"Battiti Totali: {battiti_totali}")

    # 5. Visualizzazione (Consigliata per verificare la corretta rilevazione dei picchi)
    hp.plotter(wd, m)

except Exception as e:
    print(f"Errore durante l'analisi: {e}")

# Install heartpy if not already installed

