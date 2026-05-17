import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch

# =========================================================================
# IMPOSTAZIONI GLOBALI
# =========================================================================
COLONNA_SEGNALE = "bvp"  # Colonna nativa del segnale nei tuoi file
FS = 64  # Frequenza reale del sensore BVP Empatica
FILE_OUTPUT_FINALE = "risultati_hrv_EMPATICA.csv"

# Gestione dinamica del percorso della cartella splittata
script_dir = os.path.dirname(os.path.abspath(__file__))
CARTELLA_INPUT = os.path.abspath(
    os.path.join(script_dir, "..", "data", "intervalli_splittati")
)


def estrai_numero_intervallo(filepath):
    """Funzione di supporto per ordinare i file in base al numero dell'intervallo

    (es. intervallo_1.csv, intervallo_2.csv ...

    intervallo_10.csv)
    """
    match = re.search(r"intervallo_(\d+)\.csv", os.path.basename(filepath))
    return int(match.group(1)) if match else 0


def analizza_singolo_file(path, colonna, fs):
    # 1. Caricamento dati
    try:
        df = pd.read_csv(path)
        signal = df[colonna].values
    except Exception as e:
        print(f"[{os.path.basename(path)}] Errore nel caricamento del file: {e}")
        return None

    # 2. Peak Detection (Distanza adattata a FS = 64)
    peaks, _ = find_peaks(
        signal, distance=int(fs * 0.45), prominence=np.std(signal) * 0.4
    )
    n_battiti = len(peaks)

    if n_battiti < 3:
        print(
            f"[{os.path.basename(path)}] Errore: Troppi pochi picchi rilevati."
        )
        return None

    # 3. Calcolo Intervalli RR (in millisecondi)
    rr_intervals = np.diff(peaks) * (1000.0 / fs)

    # 4. Pulizia Fisiologica Coerente
    rr_clean = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 1500)]

    if len(rr_clean) < 10:
        print(
            f"[{os.path.basename(path)}] Errore: Segnale troppo corto o sporco."
        )
        return None

    # 5. CALCOLO BPM (Logica Temporale Reale)
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

    f, psd = welch(
        rr_interp, fs=fs_interp, nperseg=min(256, len(rr_interp)), nfft=1024
    )

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


def elabora_tutti_gli_intervalli():
    # Cerca tutti i file che corrispondono al pattern 'intervallo_*.csv'
    search_pattern = os.path.join(CARTELLA_INPUT, "intervallo_*.csv")
    lista_file = glob.glob(search_pattern)

    if not lista_file:
        print(
            f"Nessun file trovato con il pattern 'intervallo_*.csv' nella cartella: {CARTELLA_INPUT}"
        )
        return

    # Ordina i file in modo sequenziale corretto (1, 2, ... 10, 11)
    lista_file = sorted(lista_file, key=estrai_numero_intervallo)

    risultati_totali = []

    print(f"Trovati {len(lista_file)} file di intervallo da elaborare...")
    print("-" * 60)

    for filepath in lista_file:
        nome_file = os.path.basename(filepath)
        print(f"Elaborazione di: {nome_file}...", end="", flush=True)

        res = analizza_singolo_file(filepath, COLONNA_SEGNALE, FS)

        if res is not None:
            risultati_totali.append(res)
            print(" Completato!")
        else:
            print(" Saltato (Errore/Dati insufficienti).")

    # Scrittura dei risultati cumulativi nel file CSV finale
    if risultati_totali:
        df_finale = pd.DataFrame(risultati_totali)

        # Il report finale viene salvato nella stessa directory in cui si trova lo script
        percorso_output = os.path.join(script_dir, FILE_OUTPUT_FINALE)
        df_finale.to_csv(percorso_output, index=False)

        print("-" * 60)
        print("ELABORAZIONE COMPLETATA CON SUCCESSO!")
        print(f"I dati combinati sono stati salvati in: {percorso_output}")
    else:
        print("-" * 60)
        print("Nessun dato valido estratto. File CSV finale non generato.")


if __name__ == "__main__":
    elabora_tutti_gli_intervalli()