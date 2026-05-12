import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

# IMPOSTAZIONI
FILE_PATH = 'ppg_200Hz_filtrato.csv'
COLONNA_SEGNALE = 'Final_Result'
FS = 200

def analizza_da_csv(path, colonna, fs):
    # 1. Caricamento dati
    try:
        df = pd.read_csv(path)
        signal = df[colonna].values
    except Exception as e:
        return f"Errore nel caricamento: {e}"

    # 2. Peak Detection
    peaks, _ = find_peaks(signal, distance=int(fs * 0.5), height=np.mean(signal))

    # 3. Calcolo Intervalli RR (in millisecondi)
    rr_intervals = np.diff(peaks) * (1000.0 / fs)

    # 4. Pulizia Fisiologica
    rr_clean = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 1500)]

    if len(rr_clean) < 10: 
        return "Errore: segnale troppo corto per analisi frequenziale."

    # 5. Calcolo Metriche Temporali
    bpm_medio = 60000.0 / np.mean(rr_clean)
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

    # 6. ANALISI FREQUENZIALE
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

    # 7. Risultati a schermo
    print("="*40)
    print(f"REPORT HRV COMPLETO - FILE: {path}")
    print("="*40)
    print(f"BPM Medio:       {bpm_medio:.2f}")
    print(f"SDNN:            {sdnn:.2f} ms")
    print(f"RMSSD:           {rmssd:.2f} ms")
    print(f"pNN50:           {pnn50:.2f} %")
    print("-" * 40)
    print(f"LF Power:        {lf_power:.2f} ms²")
    print(f"HF Power:        {hf_power:.2f} ms²")
    print(f"RAPPORTO LF/HF:  {lf_hf_ratio:.2f}")
    print("-" * 40)
    print(f"Poincaré SD1:    {sd1:.2f} ms")
    print(f"Poincaré SD2:    {sd2:.2f} ms")
    print(f"Battiti Totali:  {len(peaks)}")
    print("="*40)

    # --- 8. PULIZIA TOTALE ---
    plt.close('all') # Chiude ogni finestra rimasta appesa

    # --- 9. FINESTRA 1: Segnale e Spettro ---
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(signal, color='blue', alpha=0.6, label='PPG')
    ax1.plot(peaks, signal[peaks], "ro", label='Battiti')
    ax1.set_title("Rilevazione Battiti")
    ax1.legend()

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.fill_between(f, psd, where=lf_mask, color='orange', alpha=0.5, label='LF')
    ax2.fill_between(f, psd, where=hf_mask, color='green', alpha=0.5, label='HF')
    ax2.set_title(f"Spettro (LF/HF: {lf_hf_ratio:.2f})")
    ax2.set_xlim(0, 0.5)
    ax2.legend()
    plt.tight_layout()

    # --- 10. FINESTRA 2: Poincaré Plot (FINALMENTE DA SOLO) ---
    fig2 = plt.figure(figsize=(8, 8)) # Forza la finestra quadrata
    ax3 = fig2.add_subplot(1, 1, 1)
    
    rr_n = rr_clean[:-1]
    rr_n_plus_1 = rr_clean[1:]
    
    ax3.scatter(rr_n, rr_n_plus_1, color='purple', alpha=0.6, s=30)
    
    # QUESTO È IL SEGRETO:
    ax3.set_aspect('equal', adjustable='box') 
    
    # Centriamo il grafico sui dati per non vederlo "lontano"
    lims = [min(rr_clean)-20, max(rr_clean)+20]
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.plot(lims, lims, 'k--', alpha=0.3) # Diagonale
    
    ax3.set_title(f"Poincaré Plot (SD1: {sd1:.2f}, SD2: {sd2:.2f})")
    ax3.set_xlabel("RR_n [ms]")
    ax3.set_ylabel("RR_n+1 [ms]")
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.show()
# Esecuzione
risultato = analizza_da_csv(FILE_PATH, COLONNA_SEGNALE, FS)