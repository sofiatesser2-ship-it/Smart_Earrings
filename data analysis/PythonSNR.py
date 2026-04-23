import serial
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, medfilt

# --- CONFIGURAZIONE ---
PORTA_SERIALE = 'COM10'
BAUD_RATE = 115200
FILE_NAME = 'ppg_250Hz_filtrato.csv'
FS = 250.0  

# --- FUNZIONI DI ELABORAZIONE ---

def calcola_snr(segnale_originale, segnale_filtrato):
    """Calcola il Signal-to-Noise Ratio in dB"""
    # Rimuoviamo la media (componente DC)
    orig_clean = segnale_originale - np.mean(segnale_originale)
    filt_clean = segnale_filtrato - np.mean(segnale_filtrato)
    
    # Il rumore è la differenza tra grezzo e filtrato
    rumore = orig_clean - filt_clean
    
    potenza_segnale = np.sum(filt_clean**2)
    potenza_rumore = np.sum(rumore**2)
    
    if potenza_rumore == 0: return 0
    
    snr_db = 10 * np.log10(potenza_segnale / potenza_rumore)
    return snr_db

def applica_filtri_avanzati(data_raw):
    x = np.array(data_raw, dtype=float)
    
    # 1. NOTCH FILTER (50 Hz)
    f0 = 50.0
    Q = 30.0
    b_n, a_n = iirnotch(f0, Q, FS)
    y_notch = filtfilt(b_n, a_n, x)
    
    # 2. BUTTERWORTH BANDPASS (0.5 - 4 Hz)
    low = 0.5 / (0.5 * FS)
    high = 4.0 / (0.5 * FS)
    b_b, a_b = butter(2, [low, high], btype='band')
    y_butter = filtfilt(b_b, a_b, y_notch)
    
    # 3. MEDIAN FILTER
    y_median = medfilt(y_butter, kernel_size=7)
    
    return y_notch, y_butter, y_median

# --- LOOP DI ACQUISIZIONE ---
raw_buffer = []
time_buffer = []

try:
    ser = serial.Serial(PORTA_SERIALE, BAUD_RATE, timeout=1)
    print(f"Acquisizione a {FS}Hz avviata su {PORTA_SERIALE}. CTRL+C per terminare.")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            try:
                t, v = map(int, line.split(','))
                time_buffer.append(t)
                raw_buffer.append(v)
            except ValueError:
                continue

except KeyboardInterrupt:
    print("\n\nElaborazione in corso...")
    
    if len(raw_buffer) > 50:
        y_n, y_b, y_final = applica_filtri_avanzati(raw_buffer)
        
        # Calcolo SNR
        snr_val = calcola_snr(np.array(raw_buffer), y_final)
        print("-" * 40)
        print(f"ANALISI COMPLETATA - SNR: {snr_val:.2f} dB")
        print("-" * 40)
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Raw', 'Notch_50Hz', 'Butterworth', 'Final_Result', 'Artifact', 'SNR'])
            
            for i in range(len(raw_buffer)):
                slope = abs(raw_buffer[i] - raw_buffer[i-1]) if i > 0 else 0
                is_artifact = 1 if (slope > 200 or raw_buffer[i] > 1010 or raw_buffer[i] < 10) else 0

                writer.writerow([
                    time_buffer[i],
                    raw_buffer[i],
                    round(y_n[i], 2),
                    round(y_b[i], 2),
                    round(y_final[i], 2),
                    is_artifact,
                    round(snr_val, 2)
                ])
        print(f"Salvataggio completato in: {FILE_NAME}")
    else:
        print("Dati insufficienti.")

finally:
    if 'ser' in locals(): ser.close()

# --- GENERAZIONE GRAFICI ---
def genera_grafici():
    try:
        df = pd.read_csv(FILE_NAME)
        valore_snr = df['SNR'].iloc[0]
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Multistage PPG Signal Analysis - SNR: {valore_snr:.2f} dB', fontsize=16)

        # 1. Raw Signal
        axs[0].plot(df['Timestamp'], df['Raw'], color='gray', alpha=0.6)
        axs[0].set_title('1. Raw Signal (with Noise and Artifacts)')
        axs[0].set_ylabel('ADC Amplitude')

        # 2. After Notch
        axs[1].plot(df['Timestamp'], df['Notch_50Hz'], color='blue')
        axs[1].set_title('2. After Notch Filter (50 Hz removed)')
        axs[1].set_ylabel('Amplitude')

        # 3. After Butterworth
        axs[2].plot(df['Timestamp'], df['Butterworth'], color='orange')
        axs[2].set_title('3. After Butterworth (Beat Isolation 0.5-4 Hz)')
        axs[2].set_ylabel('Amplitude')

        # 4. Final Result
        axs[3].plot(df['Timestamp'], df['Final_Result'], color='green', linewidth=2)
        # Visualizziamo comunque i punti degli artefatti per riferimento
        artifacts = df[df['Artifact'] == 1]
        axs[3].scatter(artifacts['Timestamp'], artifacts['Final_Result'], color='red', s=10)
        
        axs[3].set_title('4. Final result (median filter)')
        axs[3].set_ylabel('Amplitude')
        
        plt.xlabel('Time (ms)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except FileNotFoundError:
        print(f"Error: the file {FILE_NAME} does not exist.")

if __name__ == "__main__":
    genera_grafici()