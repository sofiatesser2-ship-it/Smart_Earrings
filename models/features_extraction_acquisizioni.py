import pandas as pd
import numpy as np
import os
import re
from scipy.signal import welch, find_peaks
from scipy.interpolate import interp1d

# --- 1. FUNZIONI DI CALCOLO METRICHE HRV ---

def calculate_poincare_features(ibi_ms):
    if len(ibi_ms) < 2: return np.nan, np.nan
    diff_ibi = np.diff(ibi_ms)
    sd1 = np.sqrt(np.std(diff_ibi, ddof=1)**2 * 0.5)
    sd2 = np.sqrt(2 * np.std(ibi_ms, ddof=1)**2 - 0.5 * np.std(diff_ibi, ddof=1)**2)
    return sd1, sd2

def calculate_pnn50(ibi_ms):
    if len(ibi_ms) < 2: return np.nan
    diff_ibi = np.abs(np.diff(ibi_ms))
    nn50 = np.sum(diff_ibi > 50)
    return (nn50 / len(diff_ibi)) * 100

def calculate_lf_hf(ibi_ms):
    try:
        if len(ibi_ms) < 20: return np.nan
        times = np.cumsum(ibi_ms) / 1000.0
        f_interp = interp1d(times, ibi_ms, kind='cubic', fill_value="extrapolate")
        t_res = np.arange(times[0], times[-1], 0.25)
        
        signal = f_interp(t_res)
        
        # CORREZIONE: Detrendizzazione lineare anziché solo rimozione della media
        # Rimuove le derive lente che gonfiano artificialmente la banda LF
        signal = signal - np.polyval(np.polyfit(t_res, signal, 1), t_res)
        
        nperseg = min(len(signal), 256)
        f, psd = welch(signal, fs=4, nperseg=nperseg)
        
        lf = np.sum(psd[(f >= 0.04) & (f <= 0.15)])
        hf = np.sum(psd[(f >= 0.15) & (f <= 0.40)])
        
        return lf / hf if hf > 1e-6 else np.nan
    except:
        return np.nan

def clean_ibi(ibi_ms):
    # Intervallo fisiologico standard dei battiti (da 45 a 150 BPM)
    clean = ibi_ms[(ibi_ms >= 400) & (ibi_ms <= 1300)]
    return clean if len(clean) >= 20 else np.array([])

# --- 2. ELABORAZIONE DEL SEGNALE PPG DIRETTO ---

def process_ppg_file(file_path, subject_name, condition):
    """Carica il segnale PPG, pulisce i picchi dagli artefatti da movimento e assegna la Label."""
    try:
        df = pd.read_csv(file_path)
        col_target = next((c for c in df.columns if c.lower() == 'final_result'), None)
        col_time = next((c for c in df.columns if c.lower() == 'timestamp'), None)
        
        if not col_target or not col_time:
            print(f"   [Errore] Colonne richieste non trovate in {os.path.basename(file_path)}")
            return []
            
        df = df.dropna(subset=[col_time, col_target])
        
        signal = df[col_target].values
        timestamps_ms = df[col_time].values
        
    except Exception as e:
        print(f"   [Errore] Lettura fallita per {os.path.basename(file_path)}: {e}")
        return []

    if len(signal) < 1000:
        return []

    # 1. RILEVAMENTO PICCHI ADATTIVO 
    # Usiamo una prominence fissa e una distanza minima realistica (a 200Hz, 80 campioni = 400ms, max 150 BPM)
    peaks, _ = find_peaks(signal, distance=80, prominence=np.percentile(signal, 75) * 0.15)
    
    if len(peaks) < 10:
        return []

    # Interpolazione Parabolica Sub-sample per la precisione temporale
    refined_peak_times_ms = []
    fs = 200.0
    dt = 1000.0 / fs

    for p in peaks:
        if p > 0 and p < len(signal) - 1:
            y1 = signal[p-1]
            y2 = signal[p]
            y3 = signal[p+1]
            denominator = (2.0 * y2 - y1 - y3)
            shift = 0.5 * (y1 - y3) / denominator if abs(denominator) > 1e-5 else 0.0
            refined_peak_times_ms.append(timestamps_ms[p] + (shift * dt))
        else:
            refined_peak_times_ms.append(timestamps_ms[p])

    refined_peak_times_ms = np.array(refined_peak_times_ms)
    
    # 2. RIMOZIONE FILTRATA DEGLI ARTEFATTI (Outlier Protection)
    raw_ibi = np.diff(refined_peak_times_ms)
    raw_offsets_sec = (refined_peak_times_ms / 1000.0)[1:]
    
    valid_ibi = []
    valid_offsets = []
    
    for i in range(len(raw_ibi)):
        start_idx = max(0, i-4)
        end_idx = min(len(raw_ibi), i+4)
        local_median = np.median(raw_ibi[start_idx:end_idx])
        
        if (400 <= raw_ibi[i] <= 1300) and (abs(raw_ibi[i] - local_median) < 0.2 * local_median):
            valid_ibi.append(raw_ibi[i])
            valid_offsets.append(raw_offsets_sec[i])
            
    ibi_values = np.array(valid_ibi)
    ibi_offsets_sec = np.array(valid_offsets)

    if len(ibi_values) < 20:
        return []

    features = []
    step = 10  # Avanzamento di 10 secondi per entrambe le finestre

    # FINESTRE DIFFERENZIATE
    win_hrv_size = 60    # 1 minuto per le metriche temporali e geometriche
    win_lfhf_size = 120  # 2 minuti per la metrica spettrale LF_HF

    min_time = ibi_offsets_sec.min()
    max_time = ibi_offsets_sec.max()
    label = 'Baseline' if condition == 'baseline' else 'Stress'

    for sw in np.arange(min_time, max_time - win_hrv_size, step):
        # Finestra corta (1 minuto) per BPM, RMSSD, SDNN, PNN50, SD1, SD2
        mask_short = (ibi_offsets_sec >= sw) & (ibi_offsets_sec < sw + win_hrv_size)
        win_short = clean_ibi(ibi_values[mask_short])
        
        # Finestra lunga (2 minuti) per LF_HF, sincronizzata alla fine della finestra corta
        mask_long = (ibi_offsets_sec >= (sw + win_hrv_size - win_lfhf_size)) & (ibi_offsets_sec < sw + win_hrv_size)
        win_long = clean_ibi(ibi_values[mask_long])
        
        # Generiamo il record se la finestra da un minuto ha dati sufficienti
        if len(win_short) >= 15:
            bpm = 60000 / np.mean(win_short)
            rmssd = np.sqrt(np.mean(np.diff(win_short)**2))
            sdnn = np.std(win_short)
            pnn50 = calculate_pnn50(win_short)
            sd1, sd2 = calculate_poincare_features(win_short)
            
            # Calcoliamo LF_HF sui 2 minuti solo se siamo abbastanza avanti nel file da coprire l'intera finestra
            if len(win_long) >= 20:
                lf_hf = calculate_lf_hf(win_long)
            else:
                lf_hf = np.nan
            
            features.append({
                'Subject': subject_name.upper(),
                'BPM': bpm, 'RMSSD': rmssd, 'SDNN': sdnn, 
                'PNN50': pnn50, 'SD1': sd1, 'SD2': sd2,
                'LF_HF': lf_hf, 'Label': label
            })
            
    return features

# --- 3. PROCESSO PRINCIPALE ---

from pathlib import Path

if __name__ == "__main__":
    BASE_PATH = Path(__file__).resolve().parents[1]
    
    # Gestione robusta del nome della cartella (con o senza errore di battitura)
    DATA_PATH = BASE_PATH / 'acquisizoni_stress'
    if not DATA_PATH.exists():
        DATA_PATH = BASE_PATH / 'acquisizioni_stress'

    print(f"Cartella di scansione automatica: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"❌ Errore: La cartella delle acquisizioni non esiste. Verifica la sua posizione.")
        exit()

    print("Inizio scansione e ricerca file compatibili...")
    
    person_features = {}
    pattern = re.compile(r"^(s\d+)_(baseline|stress)(\d+)\.csv", re.IGNORECASE)

    for filename in os.listdir(DATA_PATH):
        match = pattern.match(filename)
        if match:
            soggetto = match.group(1).lower()
            condizione = match.group(2).lower()
            sessione = match.group(3)
            
            file_full_path = DATA_PATH / filename
            print(f"-> Analisi file PPG: {filename} (Soggetto: {soggetto.upper()} | Sessione: {sessione} | Tipo: {condizione.capitalize()})")
            
            extracted_data = process_ppg_file(str(file_full_path), soggetto, condizione)
            
            if extracted_data:
                if soggetto not in person_features:
                    person_features[soggetto] = []
                person_features[soggetto].extend(extracted_data)

    # Normalizzazione per PERSONA
    all_dfs = []
    for persona, data_list in person_features.items():
        df_person = pd.DataFrame(data_list)
        
        if df_person.empty or 'Baseline' not in df_person['Label'].values:
            print(f"⚠️ Soggetto {persona.upper()} saltato: manca del tutto la Baseline o dati insufficienti.")
            continue
            
        # Non droppiamo subito LF_HF qui per non perdere le righe che hanno solo i NaN dovuti alla finestra da 2 min
        df_person = df_person.dropna(subset=['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2'])
        cols = ['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2', 'LF_HF']
        
        person_base_means = df_person[df_person['Label'] == 'Baseline'][cols].mean()
        if person_base_means.drop('LF_HF').isnull().any():
            continue
            
        for c in cols:
            if person_base_means[c] > 0:
                df_person[c] = df_person[c] / person_base_means[c]
                
        all_dfs.append(df_person)
        print(f"✅ Normalizzazione completata con successo per il soggetto: {persona.upper()}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_filename = BASE_PATH / 'features_extraction_new_dataset.csv'
        final_df.to_csv(output_filename, index=False)
        
        print(f"\n🎉 Dataset normalizzato per persona salvato con successo!")
        print(f"➡️ '{output_filename}'")
        print(f"Record totali generati: {len(final_df)}")
        
        print("\n" + "="*65)
        print("📊 REPORT DI VERIFICA DEI DATI (MEDIE NORMALIZZATE)")
        print("="*65)
        print("💡 Linee guida per il controllo:")
        print("  - Baseline: i valori DEVONO essere uguali o vicinissimi a 1.0.")
        print("  - Stress: ci si aspetta BPM > 1.0 e metriche HRV (RMSSD, SDNN...) < 1.0.\n")
        
        metric_cols = ['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2', 'LF_HF']
        
        print("--- MEDIE GENERALI PER CONDIZIONE ---")
        print(final_df.groupby('Label')[metric_cols].mean().round(3))
        
        print("\n--- DETTAGLIO MEDIE PER SOGGETTO ---")
        print(final_df.groupby(['Subject', 'Label'])[metric_cols].mean().round(3))
        
        print("\n--- VERIFICA INTEGRITÀ E RECORD ---")
        # Conta quanti NaN ci sono (principalmente saranno nella colonna LF_HF all'inizio dei file brevi)
        print(f"🔍 Valori NaN/Nulli residui: {final_df[metric_cols].isnull().sum().to_dict()}")
        print("\nDistribuzione record per Persona:")
        print(final_df.groupby('Subject').size())
        print("\nDistribuzione complessiva per Label:")
        print(final_df.groupby('Label').size())
        print("="*65)
        
    else:
        print(f"\n❌ Errore: Nessun dato generato. Verifica che ci siano file corretti in {DATA_PATH}")