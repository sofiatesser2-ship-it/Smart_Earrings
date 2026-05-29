import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch

# IMPOSTAZIONI
CARTELLA_DATI = "./"  # Modifica se i file sono in un'altra cartella (es. 'dati/')
COLONNA_SEGNALE = "Final_Result"
FS = 200
FILE_OUTPUT_FINALE = "risultati_hrv_totale.csv"


def estrai_numero_file(filepath):
    """Funzione di supporto per ordinare i file in modo numerico naturale (1, 2, ...

    10, 11) invece che alfabetico.
    """
    match = re.search(r"140526\.(\d+)\.csv", os.path.basename(filepath))
    return int(match.group(1)) if match else 0


def analizza_singolo_file(path, colonna, fs):
    # 1. Caricamento dati
    try:
        df = pd.read_csv(path)
        signal = df[colonna].values
    except Exception as e:
        print(f"[{os.path.basename(path)}] Errore nel caricamento: {e}")
        return None

    # 2. Peak Detection
    peaks, _ = find_peaks(
        signal, distance=int(fs * 0.4), height=np.mean(signal)
    )
    n_battiti = len(peaks)

    if n_battiti < 3:
        print(
            f"[{os.path.basename(path)}] Errore: Troppi pochi picchi rilevati."
        )
        return None

    # 3. Calcolo Intervalli RR (in millisecondi)
    rr_intervals = np.diff(peaks) * (1000.0 / fs)

    # 4. Pulizia Fisiologica
    rr_clean = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 1500)]

    if len(rr_clean) < 10:
        print(
            f"[{os.path.basename(path)}] Errore: Segnale troppo corto o sporco."
        )
        return None

    # 5. CALCOLO BPM
    durata_secondi = (peaks[-1] - peaks[0]) / fs
    durata_minuti = durata_secondi / 60
    bpm_medio = (n_battiti - 1) / durata_minuti

    # 6. Calcolo Metriche Temporali (HRV)
    sdnn = np.std(rr_clean, ddof=1)
    diff_rr = np.diff(rr_clean)
    rmssd = np.sqrt(np.mean(np.square(diff_rr)))

    # pNN50
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0

    # Parametri Poincaré (SD1, SD2)
    sd_diff_rr = np.std(diff_rr, ddof=1)
    sd1 = np.sqrt(0.5 * sd_diff_rr**2)
    sd2 = np.sqrt(2 * sdnn**2 - 0.5 * sd_diff_rr**2)

    # 7. ANALISI FREQUENZIALE
    x = np.cumsum(rr_clean) / 1000.0
    x = x - x[0]

    fs_interp = 4
    f_interp = interp1d(x, rr_clean, kind="cubic")
    x_new = np.arange(0, x[-1], 1 / fs_interp)
    rr_interp = f_interp(x_new)
    rr_interp = rr_interp - np.mean(rr_interp)

    f, psd = welch(rr_interp, fs=fs_interp, nperseg=len(rr_interp), nfft=1024)

    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f > 0.15) & (f <= 0.4)

    lf_power = np.trapezoid(psd[lf_mask], f[lf_mask])
    hf_power = np.trapezoid(psd[hf_mask], f[hf_mask])
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    # Restituisce un dizionario con tutte le metriche calcolate
    return {
        "File": os.path.basename(path),
        "Battiti_Totali": n_battiti,
        "Durata_Secondi": round(durata_secondi, 1),
        "BPM_Medio": round(bpm_medio, 2),
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2),
        "pNN50_percent": round(pnn50, 2),
        "LF_Power_ms2": round(lf_power, 2),
        "HF_Power_ms2": round(hf_power, 2),
        "Rapporto_LF_HF": round(lf_hf_ratio, 2),
        "Poincare_SD1_ms": round(sd1, 2),
        "Poincare_SD2_ms": round(sd2, 2),
    }


def elabora_tutti_i_csv():
    # Cerca tutti i file che corrispondono al pattern '140526.*.csv'
    search_pattern = os.path.join(CARTELLA_DATI, "140526.*.csv")
    lista_file = glob.glob(search_pattern)

    if not lista_file:
        print(
            f"Nessun file trovato con il pattern '140526.*.csv' in '{CARTELLA_DATI}'"
        )
        return

    # Ordina i file numericamente (1, 2, 3... invece di 1, 10, 11...)
    lista_file = sorted(lista_file, key=estrai_numero_file)

    risultati_totali = []

    print(f"Trovati {len(lista_file)} file da elaborare...")
    print("-" * 50)

    for filepath in lista_file:
        nome_file = os.path.basename(filepath)
        print(f"Elaborazione di: {nome_file}...", end="", flush=True)

        res = analizza_singolo_file(filepath, COLONNA_SEGNALE, FS)

        if res is not None:
            risultati_totali.append(res)
            print(" Completato!")
        else:
            print(" Saltato (Errore/Dati non sufficienti).")

    # Scrittura dei risultati finali nel nuovo CSV
    if risultati_totali:
        df_finale = pd.DataFrame(risultati_totali)
        df_finale.to_csv(FILE_OUTPUT_FINALE, index=False)
        print("-" * 50)
        print(f"ELABORAZIONE COMPLETATA!")
        print(f"I dati di tutti i file sono stati salvati in: {FILE_OUTPUT_FINALE}")
    else:
        print("-" * 50)
        print("Nessun dato valido estratto. File CSV finale non generato.")


# Esecuzione del processo globale
if __name__ == "__main__":
    elabora_tutti_i_csv()