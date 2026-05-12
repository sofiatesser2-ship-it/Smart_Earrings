import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

# 1. Caricamento dati
path_file = 'ppg_200Hz_filtrato.csv'
df = pd.read_csv(path_file)

# Sostituisci 'signal' con il nome reale della tua colonna pre-filtrata
signal_data = df['Final_Result'].values

# 2. Parametri
fs = 200.0

try:
    # 3. Analisi Diretta
    wd, m = hp.process(signal_data, sample_rate=fs)

    # 4. Estrazione Feature
    bpm_medio = m['bpm']
    ibi_medio = m['ibi']
    sdnn = m['sdnn']
    rmssd = m['rmssd']
    
    # --- NUOVE FEATURE RICHIESTE ---
    # pNN50: Percentuale di intervalli RR adiacenti che differiscono di più di 50ms
    pnn50 = m['pnn50']
    
    # Poincaré Plot Features (SD1 e SD2)
    sd1 = m['sd1']
    sd2 = m['sd2']
    # -------------------------------

    battiti_totali = len(wd['peaklist']) - len(wd['removed_beats'])

    print(f"--- Risultati Analisi ---")
    print(f"BPM Medio:      {bpm_medio:.2f}")
    print(f"RR Medio (IBI): {ibi_medio:.2f} ms")
    print(f"SDNN:           {sdnn:.2f} ms")
    print(f"RMSSD:          {rmssd:.2f} ms")
    print(f"pNN50:          {pnn50:.2f} %")
    print(f"Poincaré SD1:   {sd1:.2f} ms (Variabilità a breve termine)")
    print(f"Poincaré SD2:   {sd2:.2f} ms (Variabilità a lungo termine)")
    print(f"Battiti Totali: {battiti_totali}")

    # 5. Visualizzazione
    # Il plotter standard di HeartPy mostra il segnale e i picchi
    hp.plotter(wd, m)
    
    # 6. Visualizzazione Poincaré Plot (Grafico dedicato)
    plt.figure(figsize=(6, 6))
    plt.title("Poincaré Plot")
    # HeartPy non ha un plotter dedicato solo per Poincaré, 
    # ma possiamo visualizzarlo se necessario o usare i dati estratti
    plt.scatter(wd['RR_list'][:-1], wd['RR_list'][1:], c='blue', alpha=0.5)
    plt.xlabel('RR_n (ms)')
    plt.ylabel('RR_n+1 (ms)')
    plt.show()

except Exception as e:
    print(f"Errore durante l'analisi: {e}")