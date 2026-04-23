import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

# IMPOSTAZIONI
FILE_PATH = 'csv 170426 (3).csv'
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

    if len(rr_clean) < 10: # Serve un minimo di dati per la frequenza
        return "Errore: segnale troppo corto per analisi frequenziale."

    # 5. Calcolo Metriche Temporali
    bpm_medio = 60000.0 / np.mean(rr_clean)
    sdnn = np.std(rr_clean, ddof=1)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_clean))))

    # --- 6. ANALISI FREQUENZIALE (LF/HF Ratio) ---
    
    # Creiamo un asse temporale cumulativo per gli intervalli RR (in secondi)
    x = np.cumsum(rr_clean) / 1000.0
    x = x - x[0] # Inizia da 0
    
    # Interpolazione a 4Hz per rendere il segnale uniforme
    fs_interp = 4 
    f_interp = interp1d(x, rr_clean, kind='cubic')
    x_new = np.arange(0, x[-1], 1/fs_interp)
    rr_interp = f_interp(x_new)
    
    # Rimozione della media (detrending semplice)
    rr_interp = rr_interp - np.mean(rr_interp)

    # Calcolo della Densità Spettrale di Potenza (PSD) 
    f, psd = welch(rr_interp, fs=fs_interp, nperseg=len(rr_interp), nfft=1024)


    # Definizione bande di frequenza
    # LF: 0.04 - 0.15 Hz | HF: 0.15 - 0.4 Hz
    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f > 0.15) & (f <= 0.4)   

     # Calcolo aree (Potenza) usando il nuovo metodo trapezoid di NumPy 2.0+
    lf_power = np.trapezoid(psd[lf_mask], f[lf_mask])
    hf_power = np.trapezoid(psd[hf_mask], f[hf_mask])
    
    # Rapporto LF/HF
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    # 7. Risultati a schermo
    print("="*40)
    print(f"REPORT HRV COMPLETO - FILE: {path}")
    print("="*40)
    print(f"BPM Medio:       {bpm_medio:.2f}")
    print(f"SDNN:            {sdnn:.2f} ms")
    print(f"RMSSD:           {rmssd:.2f} ms")
    print("-" * 40)
    print(f"LF Power:        {lf_power:.2f} ms²")
    print(f"HF Power:        {hf_power:.2f} ms²")
    print(f"RAPPORTO LF/HF:  {lf_hf_ratio:.2f}")
    print("-" * 40)
    print(f"Battiti Totali:  {len(peaks)}")
    print("="*40)

    # 8. Visualizzazione Grafica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Segnale e Picchi
    ax1.plot(signal, label='Segnale PPG', color='blue', alpha=0.6)
    ax1.plot(peaks, signal[peaks], "ro", label='Battiti')
    ax1.set_title("Rilevazione Battiti")
    ax1.legend()


    # Spettro di Frequenza
    ax2.fill_between(f, psd, where=lf_mask, color='orange', alpha=0.5, label='Low Frequency (LF)')
    ax2.fill_between(f, psd, where=hf_mask, color='green', alpha=0.5, label='High Frequency (HF)')
    ax2.legend()
    ax2.set_xlim(0, 0.5)
    ax2.set_title(f"Analisi Spettrale (PSD) - Rapporto LF/HF: {lf_hf_ratio:.2f}")
    ax2.set_xlabel("Frequenza [Hz]")
    ax2.set_ylabel("Densità Spettrale [ms²/Hz]")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return rr_clean

# Esecuzione
risultato = analizza_da_csv(FILE_PATH, COLONNA_SEGNALE, FS)


# Esecuzione
risultato = analizza_da_csv(FILE_PATH, COLONNA_SEGNALE, FS)


# Se il risultato è una stringa, significa che è scattato un messaggio di errore
if isinstance(risultato, str):
    print(f"ATTENZIONE: {risultato}")
else:
    print("Analysis Completed Successfully")


