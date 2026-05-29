import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, welch

# =========================================================================
# IMPOSTAZIONI GLOBALI
# =========================================================================
COLONNA_SEGNALE = "bvp"  # Colonna nativa del segnale nei tuoi file
FS = 64  # Frequenza reale del sensore BVP Empatica
FILE_OUTPUT_FINALE = "risultati_hrv_EMPATICA.csv"

# Gestione dinamica dei percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cartella di input: sale di un livello da 'models' e va in 'data/intervalli_splittati'
CARTELLA_INPUT = os.path.abspath(
    os.path.join(script_dir, "..", "data", "intervalli_splittati")
)

# Cartella di output: sale di un livello da 'models' per salvare direttamente in 'Smart_Earrings'
CARTELLA_OUTPUT_TARGET = os.path.abspath(os.path.join(script_dir, ".."))


def estrai_numero_intervallo(filepath):
    """Funzione di supporto per ordinare i file in base al numero dell'intervallo"""
    match = re.search(r"intervallo_(\d+)\.csv", os.path.basename(filepath))
    return int(match.group(1)) if match else 0


def filtro_passa_banda(segnale, fs, lowcut=0.5, highcut=4.0, order=4):
    """Filtra il segnale BVP per rimuovere il rumore ad alta frequenza e i trend lenti"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, segnale)


def pulisci_intervalli_rr(rr_intervals):
    """Pulisce gli intervalli RR rimuovendo gli outlier fisiologici e i salti improvvisi

    senza distruggere la sequenza temporale (sostituzione con la mediana locale)
    """
    rr_clean = rr_intervals.copy()
    # 1. Filtro fisiologico assoluto (BPM tra 40 e 150)
    invalid_mask = (rr_clean < 400) | (rr_clean > 1500)

    # 2. Filtro basato sulla deviazione dalla mediana mobile (rimozione artefatti improvvisi)
    for i in range(1, len(rr_clean)):
        if abs(rr_clean[i] - rr_clean[i - 1]) > (0.2 * rr_clean[i - 1]):
            invalid_mask[i] = True

    # Sostituiamo i valori non validi con la mediana circostante per non interrompere la sequenza temporale
    if np.any(invalid_mask):
        mediana_globale = np.median(rr_clean[~invalid_mask])
        rr_clean[invalid_mask] = mediana_globale

    return rr_clean


def analizza_singolo_file(path, colonna, fs):
    # 1. Caricamento dati
    try:
        df = pd.read_csv(path)
        signal_raw = df[colonna].values
    except Exception as e:
        print(f"[{os.path.basename(path)}] Errore nel caricamento del file: {e}")
        return None

    if len(signal_raw) < fs * 10:  # Almeno 10 secondi di segnale minimi
        return None

    # 1b. Pre-filtraggio del segnale BVP (Cruciale per i sensori al polso/orecchio)
    signal = filtro_passa_banda(signal_raw, fs)

    # 2. Peak Detection adattata alla frequenza di 64Hz
    peaks, _ = find_peaks(
        signal, distance=int(fs * 0.40), prominence=np.std(signal) * 0.5
    )
    n_battiti = len(peaks)

    if n_battiti < 10:
        print(
            f"[{os.path.basename(path)}] Errore: Troppi pochi picchi rilevati."
        )
        return None

    # 3. Calcolo Intervalli RR (in millisecondi)
    rr_intervals = np.diff(peaks) * (1000.0 / fs)

    # 4. Pulizia Fisiologica Avanzata
    rr_clean = pulisci_intervalli_rr(rr_intervals)

    if len(rr_clean) < 10:
        print(
            f"[{os.path.basename(path)}] Errore: Segnale troppo corto o sporco dopo pulizia."
        )
        return None

    # 5. Calcolo BPM reale basato sulla media degli intervalli puliti
    durata_secondi = np.sum(rr_clean) / 1000.0
    durata_minuti = durata_secondi / 60
    bpm_medio = 60000.0 / np.mean(rr_clean)

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

    # 7. Analisi Frequenziale (Risolti i bug di compressione temporale e bordi)
    x = np.cumsum(rr_clean) / 1000.0
    x = x - x[0]

    fs_interp = 4  # Campionamento standard HRV frequenziale a 4 Hz
    f_interp = interp1d(x, rr_clean, kind="cubic", fill_value="extrapolate")
    # Ci assicuriamo di restare all'interno del range corretto per evitare aliasing o crash sui bordi
    x_new = np.arange(0, x[-1] - (1 / fs_interp), 1 / fs_interp)

    if len(x_new) < 16:  # Finestra minima per l'algoritmo di Welch
        return None

    rr_interp = f_interp(x_new)
    rr_interp = rr_interp - np.mean(rr_interp)  # Detrending lineare di base

    nperseg = min(256, len(rr_interp))
    f, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg, nfft=1024)

    # Maschere frequenziali standard (Tarvainen et al.)
    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f > 0.15) & (f <= 0.4)

    # Integrazione delle potenze dello spettro
    lf_power = np.trapz(psd[lf_mask], f[lf_mask]) if any(lf_mask) else 0
    hf_power = np.trapz(psd[hf_mask], f[hf_mask]) if any(hf_mask) else 0
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

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
    search_pattern = os.path.join(CARTELLA_INPUT, "intervallo_*.csv")
    lista_file = glob.glob(search_pattern)

    if not lista_file:
        print(
            f"Nessun file trovato con il pattern 'intervallo_*.csv' nella cartella: {CARTELLA_INPUT}"
        )
        return

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

    if risultati_totali:
        df_finale = pd.DataFrame(risultati_totali)

        # Costruzione del percorso di output forzato nella cartella principale 'Smart_Earrings'
        percorso_output = os.path.join(
            CARTELLA_OUTPUT_TARGET, FILE_OUTPUT_FINALE
        )
        df_finale.to_csv(percorso_output, index=False)

        print("-" * 60)
        print("ELABORAZIONE COMPLETATA CON SUCCESSO!")
        print(f"I dati combinati sono stati salvati in: {percorso_output}")
    else:
        print("-" * 60)
        print("Nessun dato valido estratto. File CSV finale non generato.")


if __name__ == "__main__":
    # Gestione delle differenze di versione per la funzione di integrazione numerica di numpy
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid

    elabora_tutti_gli_intervalli()