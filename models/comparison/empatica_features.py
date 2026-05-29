import os
from datetime import datetime
import numpy as np
import pandas as pd

print("=" * 60)
print(" GENERAZIONE CSV INDIVIDUALI PER OGNI INTERVALLO DI 2 MINUTI ")
print("=" * 60)

# 1. I TUOI 16 INTERVALLI PRECISI
intervalli_input = {
    "1": ("13.12.58", "13.14.58"),
    "2": ("13.33.31", "13.35.31"),
    "3": ("15.34.35", "15.36.35"),
    "4": ("15.40.59", "15.42.59"),
    "5": ("15.53.04", "15.55.04"),
    "6": ("15.57.05", "15.59.05"),
    "7": ("16.01.48", "16.03.48"),
    "8": ("16.05.46", "16.07.46"),
    "9": ("16.10.52", "16.12.52"),
    "10": ("16.16.05", "16.18.05"),
    "11": ("16.28.38", "16.30.28"),
    "12": ("16.34.02", "16.36.02"),
    "13": ("16.38.04", "16.40.04"),
    "14": ("16.49.50", "16.51.50"),
    "15": ("17.02.12", "17.04.12"),
    "16": ("17.12.58", "17.14.58"),
}

# 2. DEFINIZIONE PERCORSI (Leggiamo bvp_raw per avere la massima densità temporale)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_bvp = os.path.abspath(os.path.join(script_dir, "..", "data", "bvp_raw.parquet"))
output_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "intervalli_splittati"))

# Creiamo la cartella di destinazione per non fare disordine
os.makedirs(output_dir, exist_ok=True)

# 3. CARICAMENTO DATI ORIGINALI
try:
    df = pd.read_parquet(file_path_bvp)
    print("File dei dati originari caricato con successo!")
except FileNotFoundError:
    # Se non trova il bvp_raw, prova in automatico con il file dei picchi sistolici
    file_path_peaks = os.path.abspath(os.path.join(script_dir, "..", "data", "systolic_peaks_raw.parquet"))
    try:
        df = pd.read_parquet(file_path_peaks)
        print("File dei picchi sistolici caricato come sorgente alternativa!")
    except FileNotFoundError:
        print("[ERRORE] File non trovati in data/. Verifica la presenza di bvp_raw.parquet o systolic_peaks_raw.parquet.")
        exit()

if df.index.name is not None:
    df = df.reset_index()

# Individuazione automatica della colonna del tempo
timestamp_col = [c for c in df.columns if "time" in c.lower() or "utc" in c.lower() or c == "index"][0]

# Allineamento robusto del fuso orario (da UTC a locale italiano Europe/Rome)
df[timestamp_col] = pd.to_datetime(df[timestamp_col])
if df[timestamp_col].dt.tz is None:
    df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
df[timestamp_col] = df[timestamp_col].dt.tz_convert("Europe/Rome")

df = df.sort_values(by=timestamp_col)
df["ora_stringa"] = df[timestamp_col].dt.strftime("%H:%M:%S")

print(f"Orari totali disponibili nella registrazione: {df['ora_stringa'].iloc[0]} -> {df['ora_stringa'].iloc[-1]}\n")

# 4. PARZIALIZZAZIONE E SCRITTURA DEI CSV INDIVIDUALI
files_creati = 0

for id_intervallo, (ora_inizio, ora_fine) in intervalli_input.items():
    t_start = ora_inizio.replace(".", ":")
    t_end = ora_fine.replace(".", ":")
    
    # Taglio chirurgico del dataframe sull'orario esatto
    df_segmento = df[(df["ora_stringa"] >= t_start) & (df["ora_stringa"] <= t_end)].copy()
    
    if len(df_segmento) == 0:
        # Se l'orario è fuori dal file attuale (es. dopo le 16:04), lo script salta l'intervallo senza bloccarsi
        continue
        
    # Eliminiamo la colonna ausiliaria per lasciarti il file pulito al 100%
    df_output = df_segmento.drop(columns=["ora_stringa"], errors="ignore")
    
    # Prepariamo un nome file chiaro ed esplicito
    nome_clean_inizio = ora_inizio.replace(".", "")
    nome_clean_fine = ora_fine.replace(".", "")
    file_nome = f"intervallo_{id_intervallo}.csv"    
    percorso_salvataggio = os.path.join(output_dir, file_nome)
    
    # Esportazione in CSV
    df_output.to_csv(percorso_salvataggio, index=False)
    print(f" -> Creato CSV Intervallo {id_intervallo} ({len(df_output)} righe) -> data/intervalli_splittati/{file_nome}")
    files_creati += 1

print("\n" + "=" * 60)
print(f" OPERAZIONE COMPLETATA CON SUCCESSO! Generati {files_creati} file CSV.")
print(f" Cartella di destinazione: {output_dir}")
print("=" * 60)