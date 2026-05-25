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
        signal = signal - np.mean(signal)
        
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
    """Carica il segnale PPG, trova i picchi e assegna la Label (Baseline o Stress)."""
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
        timestamps_sec = timestamps_ms / 1000.0
        
    except Exception as e:
        print(f"   [Errore] Lettura fallita per {os.path.basename(file_path)}: {e}")
        return []

    if len(signal) < 1000:
        return []

    # Trova i picchi del segnale PPG (frequenza campionamento 200Hz)
    peaks, _ = find_peaks(signal, distance=80, prominence=np.std(signal)*0.2)
    
    if len(peaks) < 10:
        return []

    # Genera i veri Inter-Beat Intervals (IBI) in millisecondi
    peak_times_ms = timestamps_ms[peaks]
    peak_times_sec = timestamps_sec[peaks]
    
    ibi_values = np.diff(peak_times_ms)
    ibi_offsets_sec = peak_times_sec[1:]

    features = []
    window_size = 120  # Finestra mobile di 2 minuti
    step = 10         # Avanzamento di 10 secondi

    min_time = ibi_offsets_sec.min()
    max_time = ibi_offsets_sec.max()

    label = 'Baseline' if condition == 'baseline' else 'Stress'

    for sw in np.arange(min_time, max_time - window_size, step):
        mask = (ibi_offsets_sec >= sw) & (ibi_offsets_sec < sw + window_size)
        win = ibi_values[mask]
        
        win = clean_ibi(win)
        
        if len(win) >= 20:
            bpm = 60000 / np.mean(win)
            rmssd = np.sqrt(np.mean(np.diff(win)**2))
            sdnn = np.std(win)
            lf_hf = calculate_lf_hf(win)
            pnn50 = calculate_pnn50(win)
            sd1, sd2 = calculate_poincare_features(win)
            
            features.append({
                'Subject': subject_name.upper(), # Es: "S1", "S2"
                'BPM': bpm, 'RMSSD': rmssd, 'SDNN': sdnn, 
                'PNN50': pnn50, 'SD1': sd1, 'SD2': sd2,
                'LF_HF': lf_hf, 'Label': label
            })
            
    return features

# --- 3. PROCESSO PRINCIPALE ---

if __name__ == "__main__":
    # Rileva la cartella dove si trova questo script
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Risale alla cartella principale 'Smart_Earrings' e punta a 'acquisizioni_stress'
    BASE_PATH = os.path.dirname(CURRENT_DIR) 
    DATA_PATH = os.path.join(BASE_PATH, 'acquisizioni_stress')

    print(f"Cartella di scansione automatica: {DATA_PATH}")
    
    # Controllo di sicurezza se la cartella esiste effettivamente
    if not os.path.exists(DATA_PATH):
        print(f"❌ Errore: La cartella '{DATA_PATH}' non esiste. Verifica la sua posizione.")
        exit()

    print("Inizio scansione e ricerca file compatibili...")
    
    # Dizionario principale raggruppato per PERSONA (chiavi: 's1', 's2', ecc.)
    person_features = {}
    pattern = re.compile(r"^(s\d+)_(baseline|stress)(\d+)\.csv", re.IGNORECASE)

    for filename in os.listdir(DATA_PATH):
        match = pattern.match(filename)
        if match:
            soggetto = match.group(1).lower()
            condizione = match.group(2).lower()
            sessione = match.group(3)
            
            file_full_path = os.path.join(DATA_PATH, filename)
            print(f"-> Analisi file PPG: {filename} (Soggetto: {soggetto.upper()} | Sessione: {sessione} | Tipo: {condizione.capitalize()})")
            
            extracted_data = process_ppg_file(file_full_path, soggetto, condizione)
            
            if extracted_data:
                if soggetto not in person_features:
                    person_features[soggetto] = []
                person_features[soggetto].extend(extracted_data)

    # Normalizzazione per PERSONA (Rispetto alla Baseline complessiva del soggetto)
    all_dfs = []
    for persona, data_list in person_features.items():
        df_person = pd.DataFrame(data_list)
        
        # Verifica che per questa persona ci sia almeno un file di Baseline
        if df_person.empty or 'Baseline' not in df_person['Label'].values:
            print(f"⚠️ Soggetto {persona.upper()} saltato: manca del tutto la Baseline o dati insufficienti.")
            continue
            
        df_person = df_person.dropna(subset=['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2'])
        cols = ['BPM', 'RMSSD', 'SDNN', 'PNN50', 'SD1', 'SD2', 'LF_HF']
        
        # Calcola la media delle feature aggregando TUTTE le sessioni di Baseline di QUESTO soggetto
        person_base_means = df_person[df_person['Label'] == 'Baseline'][cols].mean()
        if person_base_means.isnull().any():
            continue
            
        # Normalizzazione: divide ogni record per la media di Baseline del soggetto
        for c in cols:
            if person_base_means[c] > 0:
                df_person[c] = df_person[c] / person_base_means[c]
                
        all_dfs.append(df_person.dropna())
        print(f"✅ Normalizzazione completata con successo per il soggetto: {persona.upper()}")

    # Salvataggio complessivo sul file finale
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Salva il dataset risultante dentro la cartella principale 'Smart_Earrings'
        output_filename = os.path.join(BASE_PATH, 'features_extraction_new_dataset.csv')
        final_df.to_csv(output_filename, index=False)
        print(f"\n🎉 Dataset normalizzato per persona salvato con successo!")
        print(f"➡️ '{output_filename}'")
        print(f"Record totali generati: {len(final_df)}")
        print("\nDistribuzione record per Persona:")
        print(final_df.groupby('Subject').size())
        print("\nDistribuzione complessiva per Label:")
        print(final_df.groupby('Label').size())
    else:
        print(f"\n❌ Errore: Nessun dato generato. Verifica che ci siano file corretti in {DATA_PATH}")