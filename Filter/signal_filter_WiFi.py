import socket
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, medfilt
import traceback

# --- CONFIGURAZIONE ---
UDP_IP = "0.0.0.0"       # Ascolta su tutte le reti
UDP_PORT = 5005              
FILE_NAME = 'ppg_200Hz_filtrato.csv'
FS = 200.0  

# --- FUNZIONI DI ELABORAZIONE ---
def calcola_snr(segnale_originale, segnale_filtrato):
    orig_clean = segnale_originale - np.mean(segnale_originale)
    filt_clean = segnale_filtrato - np.mean(segnale_filtrato)
    rumore = orig_clean - filt_clean
    potenza_segnale = np.sum(filt_clean**2)
    potenza_rumore = np.sum(rumore**2)
    if potenza_rumore == 0: return 0
    return 10 * np.log10(potenza_segnale / potenza_rumore)

def applica_filtri_avanzati(data_raw):
    x = np.array(data_raw, dtype=float)
    # Notch
    b_n, a_n = iirnotch(50.0, 30.0, FS)
    y_notch = filtfilt(b_n, a_n, x)
    # Butterworth
    low = 0.5 / (0.5 * FS)
    high = 4.0 / (0.5 * FS)
    b_b, a_b = butter(2, [low, high], btype='band')
    y_butter = filtfilt(b_b, a_b, y_notch)
    # Median
    y_median = medfilt(y_butter, kernel_size=7)
    return y_notch, y_butter, y_median

# --- LOOP DI ACQUISIZIONE ---
raw_buffer = []
time_buffer = []

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"In ascolto per dati UDP sull'IP {UDP_IP} alla porta {UDP_PORT}...")
    print(f"Acquisizione a {FS}Hz avviata. Premi CTRL+C per terminare.")

    # Questo è il ciclo infinito che lo tiene acceso
    while True:
        data, addr = sock.recvfrom(1024)
        line = data.decode('utf-8', errors='ignore').strip()
        
        if line:
            try:
                t_str, v_str = line.split(',')
                t, v = int(t_str), int(v_str)
                time_buffer.append(t)
                raw_buffer.append(v)
                print(f"t={t} ms  value={v}")
            except ValueError:
                continue

except KeyboardInterrupt:
    print("\n\nAcquisizione interrotta volontariamente. Elaborazione in corso...")
    if len(raw_buffer) > 50:
        y_n, y_b, y_final = applica_filtri_avanzati(raw_buffer)
        snr_val = calcola_snr(np.array(raw_buffer), y_final)
        print(f"ANALISI COMPLETATA - SNR: {snr_val:.2f} dB")
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Raw', 'Notch_50Hz', 'Butterworth', 'Final_Result', 'Artifact', 'SNR'])
            for i in range(len(raw_buffer)):
                slope = abs(raw_buffer[i] - raw_buffer[i-1]) if i > 0 else 0
                is_artifact = 1 if (slope > 200 or raw_buffer[i] > 1010 or raw_buffer[i] < 10) else 0
                writer.writerow([time_buffer[i], raw_buffer[i], round(y_n[i], 2), round(y_b[i], 2), round(y_final[i], 2), is_artifact, round(snr_val, 2)])
        print(f"Salvataggio completato in: {FILE_NAME}")
    else:
        print("Dati insufficienti per l'elaborazione.")

except Exception as e:
    # SE CRASHA DI COLPO, STAMPERA' L'ERRORE QUI
    print("\n" + "!"*50)
    print("ERRORE CRITICO! IL PROGRAMMA SI È FERMATO PERCHÉ:")
    print(e)
    traceback.print_exc()
    print("!"*50 + "\n")

finally:
    if 'sock' in locals(): sock.close()

# --- GENERAZIONE GRAFICI ---
def genera_grafici():
    try:
        df = pd.read_csv(FILE_NAME)
        valore_snr = df['SNR'].iloc[0]
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Multistage PPG Signal Analysis - SNR: {valore_snr:.2f} dB', fontsize=16)

        axs[0].plot(df['Timestamp'], df['Raw'], color='gray', alpha=0.6)
        axs[0].set_title('1. Raw Signal')
        axs[1].plot(df['Timestamp'], df['Notch_50Hz'], color='blue')
        axs[1].set_title('2. Notch Filter')
        axs[2].plot(df['Timestamp'], df['Butterworth'], color='orange')
        axs[2].set_title('3. Butterworth Bandpass')
        axs[3].plot(df['Timestamp'], df['Final_Result'], color='green', linewidth=2)
        
        artifacts = df[df['Artifact'] == 1]
        axs[3].scatter(artifacts['Timestamp'], artifacts['Final_Result'], color='red', s=10)
        axs[3].set_title('4. Final Result')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    if len(raw_buffer) > 50:
        genera_grafici()