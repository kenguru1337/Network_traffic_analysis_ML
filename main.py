import scapy.all as scapy
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import time


# Функция для сбора сетевого трафика
def capture_traffic(duration=60, output_file="traffic.csv"):
    packets = scapy.sniff(timeout=duration)
    data = []

    for packet in packets:
        if packet.haslayer(scapy.IP):
            data.append([
                packet.time,
                len(packet),
                packet[scapy.IP].src,
                packet[scapy.IP].dst,
                packet[scapy.IP].proto
            ])

    df = pd.DataFrame(data, columns=["timestamp", "length", "src", "dst", "protocol"])
    df.to_csv(output_file, index=False)
    print(f"Traffic captured and saved to {output_file}")


# Функция для подготовки данных
def preprocess_data(input_file="traffic.csv", output_file="processed.csv"):
    df = pd.read_csv(input_file)
    df["protocol"] = df["protocol"].astype(str)  # Преобразуем числовой протокол в строку
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)

    df_numeric = df.drop(columns=["src", "dst", "timestamp"])  # Исключаем строковые данные
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    joblib.dump(scaler, "scaler.pkl")

    pd.DataFrame(df_scaled).to_csv(output_file, index=False)
    print(f"Data preprocessed and saved to {output_file}")


# Функция для обучения нейросети
def train_model(input_file="processed.csv", model_file="model.h5"):
    df = pd.read_csv(input_file)
    X_train = df.values

    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(X_train.shape[1], activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=10, batch_size=32, verbose=1)

    model.save(model_file)
    print(f"Model trained and saved to {model_file}")


# Функция для обнаружения аномалий
def detect_anomalies(model_file="model.h5", scaler_file="scaler.pkl", input_file="processed.csv"):
    model = tf.keras.models.load_model(model_file)
    scaler = joblib.load(scaler_file)
    df = pd.read_csv(input_file)
    X_test = df.values

    reconstructions = model.predict(X_test)
    mse = np.mean(np.abs(reconstructions - X_test), axis=1)
    threshold = np.percentile(mse, 95)  # Устанавливаем порог аномалий

    anomalies = mse > threshold
    anomaly_indices = np.where(anomalies)[0]
    print(f"Detected {len(anomaly_indices)} anomalies")

    return anomaly_indices


if __name__ == "__main__":
    capture_traffic(duration=60)  # Собираем трафик 60 секунд
    preprocess_data()
    train_model()
    anomalies = detect_anomalies()
    print("Anomalies detected at indices:", anomalies)
