import socket
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, medfilt
import traceback

# --- CONFIGURAZIONE ---
UDP_IP = "0.0.0.0"       # Ascolta su tutte le interfacce di rete
UDP_PORT = 5005              
FILE_NAME = 'ppg_200Hz_filtrato.csv'
FS = 200.0  

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
    # Filtro Notch (rimuove rumore di rete 50Hz)
    b_n, a_n = iirnotch(50.0, 30.0, FS)
    y_notch = filtfilt(b_n, a_n, x)
    # Filtro Butterworth Bandpass (0.5 - 4.0 Hz per battito cardiaco)
    low = 0.5 / (0.5 * FS)
    high = 4.0 / (0.5 * FS)
    b_b, a_b = butter(2, [low, high], btype='band')
    y_butter = filtfilt(b_b, a_b, y_notch)
    # Filtro Mediano (rimuove spike improvvisi)
    y_median = medfilt(y_butter, kernel_size=7)
    return y_notch, y_butter, y_median

raw_buffer = []
time_buffer = []

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    # Timeout di 5 secondi se non riceve nulla
    sock.settimeout(5.0) 
    
    print(f"In ascolto sulla porta {UDP_PORT}...")
    print("In attesa di dati dall'Arduino (assicurati che sia CONNESSO al WiFi)...")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8', errors='ignore').strip()
            
            if line and ',' in line:
                t_str, v_str = line.split(',')
                t, v = int(t_str), int(v_str)
                time_buffer.append(t)
                raw_buffer.append(v)
                print(f"Ricevuto -> Tempo: {t} ms | Valore: {v}")
        except socket.timeout:
            print("...nessun dato ricevuto negli ultimi 5 secondi...")

except KeyboardInterrupt:
    print("\nAcquisizione interrotta. Elaborazione dati...")
    if len(raw_buffer) > 50:
        y_n, y_b, y_final = applica_filtri_avanzati(raw_buffer)
        snr_val = calcola_snr(np.array(raw_buffer), y_final)
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Raw', 'Notch_50Hz', 'Butterworth', 'Final_Result', 'Artifact', 'SNR'])
            for i in range(len(raw_buffer)):
                slope = abs(raw_buffer[i] - raw_buffer[i-1]) if i > 0 else 0
                is_artifact = 1 if (slope > 200 or raw_buffer[i] > 4000 or raw_buffer[i] < 10) else 0
                writer.writerow([time_buffer[i], raw_buffer[i], round(y_n[i], 2), round(y_b[i], 2), round(y_final[i], 2), is_artifact, round(snr_val, 2)])
        print(f"Analisi completata! SNR: {snr_val:.2f} dB. File salvato.")
        
        # Mostra i grafici
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(time_buffer, raw_buffer, color='gray')
        axs[0].set_title('Raw Signal')
        axs[1].plot(time_buffer, y_n, color='blue')
        axs[1].set_title('Notch Filter')
        axs[2].plot(time_buffer, y_b, color='orange')
        axs[2].set_title('Butterworth Filter')
        axs[3].plot(time_buffer, y_final, color='green')
        axs[3].set_title('Final Filtered Signal')
        plt.tight_layout()
        plt.show()
    else:
        print("Dati insufficienti per generare grafici.")

except Exception as e:
    print(f"ERRORE: {e}")
    traceback.print_exc()
finally:
    sock.close()