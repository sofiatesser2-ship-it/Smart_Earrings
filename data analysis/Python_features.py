import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d
from pathlib import Path

# IMPOSTAZIONI
FILE_PATH = 's1_baseline1.csv'
COLONNA_SEGNALE = 'Final_Result'
FS = 200

def analizza_da_csv(path, colonna, fs):
    # 1. Caricamento dati
    try:
        script_dir = Path(__file__).resolve().parent
        percorso_completo = script_dir.parent / 'acquisizioni_stress' / path
        
        # Spostiamo il print all'inizio assoluto del try per essere sicuri che lo legga
        print(f"--> Tentativo di apertura: {percorso_completo}")
        
        df = pd.read_csv(percorso_completo)
        signal = df[colonna].values
    except Exception as e:
        return f"Errore nel caricamento: {e}"

    # 2. Peak Detection
    # Regoliamo la distanza minima tra picchi (es. 0.4s = max 150 BPM)
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4), height=np.mean(signal))
    n_battiti = len(peaks)

    # 3. Calcolo Intervalli RR (in millisecondi)
    rr_intervals = np.diff(peaks) * (1000.0 / fs)

    # 4. Pulizia Fisiologica (rimozione outlier per HRV)
    rr_clean = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 1500)]

    if len(rr_clean) < 10: 
        return "Errore: segnale troppo corto o troppo sporco per analisi."

    # 5. CALCOLO BPM (Logica Temporale Reale)
    # Calcoliamo la durata basandoci sui campioni tra il primo e l'ultimo picco
    durata_secondi = (peaks[-1] - peaks[0]) / fs
    durata_minuti = durata_secondi / 60
    
    # Il BPM corretto si calcola come (numero intervalli) / minuti
    bpm_medio = (n_battiti - 1) / durata_minuti

    # 6. Calcolo Metriche Temporali (HRV)
    sdnn = np.std(rr_clean, ddof=1)
    diff_rr = np.diff(rr_clean)
    rmssd = np.sqrt(np.mean(np.square(diff_rr)))
    
    # pNN50
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100

    # Parametri Poincaré (SD1, SD2)
    sd_diff_rr = np.std(diff_rr, ddof=1)
    sd1 = np.sqrt(0.5 * sd_diff_rr**2)
    sd2 = np.sqrt(2 * sdnn**2 - 0.5 * sd_diff_rr**2)

    # 7. ANALISI FREQUENZIALE
    x = np.cumsum(rr_clean) / 1000.0
    x = x - x[0] 
    
    fs_interp = 4 
    f_interp = interp1d(x, rr_clean, kind='cubic')
    x_new = np.arange(0, x[-1], 1/fs_interp)
    rr_interp = f_interp(x_new)
    rr_interp = rr_interp - np.mean(rr_interp)

    f, psd = welch(rr_interp, fs=fs_interp, nperseg=len(rr_interp), nfft=1024)

    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f > 0.15) & (f <= 0.4)   

    lf_power = np.trapezoid(psd[lf_mask], f[lf_mask])
    hf_power = np.trapezoid(psd[hf_mask], f[hf_mask])
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    # 8. Risultati a schermo
    print("="*40)
    print(f"REPORT HRV COMPLETO - FILE: {path}")
    print("="*40)
    print(f"Battiti Totali:  {n_battiti}")
    print(f"Durata Analisi:  {durata_secondi:.1f} secondi ({durata_minuti:.2f} min)")
    print(f"BPM MEDIO:       {bpm_medio:.2f}")
    print("-" * 40)
    print(f"SDNN (Variab.):  {sdnn:.2f} ms")
    print(f"RMSSD (Vago):    {rmssd:.2f} ms")
    print(f"pNN50:           {pnn50:.2f} %")
    print("-" * 40)
    print(f"LF Power:        {lf_power:.2f} ms²")
    print(f"HF Power:        {hf_power:.2f} ms²")
    print(f"RAPPORTO LF/HF:  {lf_hf_ratio:.2f}")
    print("-" * 40)
    print(f"Poincaré SD1:    {sd1:.2f} ms")
    print(f"Poincaré SD2:    {sd2:.2f} ms")
    print("="*40)

    # --- 9. GRAFICI ---
    plt.close('all')

    # FINESTRA 1: Segnale e Spettro
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(signal, color='blue', alpha=0.5, label='Segnale PPG')
    ax1.scatter(peaks, signal[peaks], color='red', s=20, label='Picchi Rilevati')
    ax1.set_title(f"Rilevazione Picchi (Totale: {n_battiti})")
    ax1.set_xlabel("Campioni")
    ax1.legend()

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.fill_between(f, psd, where=lf_mask, color='orange', alpha=0.5, label='LF (Simpatico)')
    ax2.fill_between(f, psd, where=hf_mask, color='green', alpha=0.5, label='HF (Vagale)')
    ax2.set_title(f"Analisi Frequenziale (LF/HF: {lf_hf_ratio:.2f})")
    ax2.set_xlim(0, 0.5)
    ax2.set_xlabel("Frequenza [Hz]")
    ax2.legend()
    plt.tight_layout()

    # FINESTRA 2: Poincaré Plot
    fig2 = plt.figure(figsize=(7, 7))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.scatter(rr_clean[:-1], rr_clean[1:], color='purple', alpha=0.6, s=30)
    ax3.set_aspect('equal', adjustable='box') 
    lims = [0, 1000]
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.plot(lims, lims, 'k--', alpha=0.3)
    ax3.set_title("Poincaré Plot")
    ax3.set_xlabel("RR_n [ms]"); ax3.set_ylabel("RR_n+1 [ms]")
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.show()

# Esecuzione
analizza_da_csv(FILE_PATH, COLONNA_SEGNALE, FS)