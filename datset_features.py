import pandas as pd
import numpy as np
import os
from scipy.signal import welch
from scipy.interpolate import interp1d

# --- 1. FUNZIONI DI PARSING E TECNICHE ---
def get_subject_times_split(quest_path):
    try:
        df_q = pd.read_csv(quest_path, sep=None, engine='python', header=None).astype(str)
        df_q = df_q.map(lambda x: x.strip().replace(',', '.').upper() if x else "")
        rows = df_q[0].values
        start_idx = next(i for i, v in enumerate(rows) if 'START' in v)
        end_idx = next(i for i, v in enumerate(rows) if 'END' in v)
        headers = df_q.iloc[1].values
        base_col = next(i for i, v in enumerate(headers) if 'BASE' in v)
        tsst_col = next(i for i, v in enumerate(headers) if 'TSST' in v)

        def to_sec(val):
            f = float(val)
            return int(f)*60 + round((f-int(f))*100)

        tsst_start = to_sec(df_q.iloc[start_idx, tsst_col])
        tsst_end = to_sec(df_q.iloc[end_idx, tsst_col])
        mid_point = tsst_start + (tsst_end - tsst_start) // 2

        return {
            'Baseline': (to_sec(df_q.iloc[start_idx, base_col]), to_sec(df_q.iloc[end_idx, base_col])),
            'Social_Stress': (tsst_start, mid_point),
            'Cognitive_Stress': (mid_point, tsst_end)
        }
    except:
        return None

def calculate_lf_hf(ibi_ms):
    try:
        if len(ibi_ms) < 20: return np.nan # Abbassato per coerenza con soglia 20
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
    # MODIFICA: Allargati i range (300-1600) e ridotta soglia minima (20)
    clean = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1600)]
    return clean if len(clean) >= 20 else np.array([])

# --- 2. ESTRAZIONE ---
def extract_features_complete(subject_id, base_path):
    sub_folder = os.path.join(base_path, subject_id)
    ibi_p = os.path.join(sub_folder, f"{subject_id}_E4_Data", "IBI.csv")
    quest_p = os.path.join(sub_folder, f"{subject_id}_quest.csv")
    
    if not os.path.exists(ibi_p) or not os.path.exists(quest_p): return None
    tasks = get_subject_times_split(quest_p)
    if not tasks: return None

    df_ibi = pd.read_csv(ibi_p, skiprows=1, names=['offset', 'ibi'])
    df_ibi['bpm_tmp'] = 60 / df_ibi['ibi']
    
    # Sincronizzazione
    peak_time = df_ibi.loc[df_ibi['bpm_tmp'].rolling(50).mean().idxmax(), 'offset']
    sync_shift = peak_time - (tasks['Social_Stress'][0] + 150)

    features = []
    window_size = 120 
    step = 10         

    for label, (start, end) in tasks.items():
        s_f, e_f = start + sync_shift, end + sync_shift
        
        # DEBUG AGGIUNTO
        print(f"DEBUG: Soggetto {subject_id} - Task: {label} | Range: {s_f:.1f} - {e_f:.1f}")
        
        if (e_f - s_f) < window_size:
            print(f"   -> SALTATO: {label} troppo breve per {subject_id}")
            continue

        for sw in np.arange(s_f, e_f - window_size, step): 
            win = df_ibi[(df_ibi['offset'] >= sw) & (df_ibi['offset'] < sw + window_size)]['ibi'].values * 1000
            win = clean_ibi(win)
            
            # MODIFICA: Soglia >= 20
            if len(win) >= 20:
                bpm = 60000 / np.mean(win)
                rmssd = np.sqrt(np.mean(np.diff(win)**2))
                sdnn = np.std(win)
                lf_hf = calculate_lf_hf(win)
                
                features.append({
                    'Subject': subject_id, 'BPM': bpm, 'RMSSD': rmssd, 
                    'SDNN': sdnn, 'LF_HF': lf_hf, 'Label': label
                })
    
    df = pd.DataFrame(features)
    if df.empty or 'Baseline' not in df['Label'].values: return None

    # Normalizzazione
    df = df.dropna(subset=['BPM', 'RMSSD', 'SDNN'])
    cols = ['BPM', 'RMSSD', 'SDNN', 'LF_HF']
    base_means = df[df['Label'] == 'Baseline'][cols].mean()
    
    if base_means.isnull().any(): return None
    
    for c in cols:
        if base_means[c] > 0:
            df[c] = df[c] / base_means[c]
            
    return df.dropna()

# --- 3. MAIN ---
if __name__ == "__main__":
    BASE_PATH = r"C:\Users\arima\Desktop\Progetto\WESAD"
    subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
    all_dfs = []

    print("Inizio estrazione (Step 10s)...")
    for s in subjects:
        df_s = extract_features_complete(s, BASE_PATH)
        if df_s is not None:
            all_dfs.append(df_s)
            print(f"Soggetto {s} estratto correttamente.")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv('wesad_complete_ratio.csv', index=False)
        print("\nDataset salvato: 'wesad_complete_ratio.csv'")
        print(f"Righe totali: {len(final_df)}")
        print(final_df.groupby('Label').size())
    else:
        print("Nessun dato estratto.")