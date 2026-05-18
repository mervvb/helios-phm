"""
LSTM RUL Tahmini — LSTM_RUL_Tahmini.ipynb pipeline
train_FD001.txt / test_FD001.txt / RUL_FD001.txt üzerinde çalışır.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Sabit kolon isimleri (notebook ile aynı) ──────────────────────────────
COLUMN_NAMES = [
    "unit_number", "time_cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]
RUL_CAP    = 125
WINDOW_SIZE = 30


# ── Yardımcı fonksiyonlar (notebook Hücre 4-5) ────────────────────────────

def create_sequences(data, feature_cols, target_col, window_size):
    X, y = [], []
    for unit in data["unit_number"].unique():
        unit_data = data[data["unit_number"] == unit].sort_values("time_cycle")
        feature_values = unit_data[feature_cols].values
        target_values  = unit_data[target_col].values
        for i in range(window_size, len(unit_data)):
            X.append(feature_values[i - window_size:i])
            y.append(target_values[i])
    return np.array(X), np.array(y)


def create_test_sequences(test_data, feature_cols, window_size):
    X_test, unit_numbers, last_cycles = [], [], []
    for unit in test_data["unit_number"].unique():
        unit_data      = test_data[test_data["unit_number"] == unit].sort_values("time_cycle")
        feature_values = unit_data[feature_cols].values
        if len(unit_data) >= window_size:
            sequence = feature_values[-window_size:]
        else:
            pad_size = window_size - len(unit_data)
            pad      = np.repeat(feature_values[0:1], pad_size, axis=0)
            sequence = np.vstack([pad, feature_values])
        X_test.append(sequence)
        unit_numbers.append(unit)
        last_cycles.append(unit_data["time_cycle"].iloc[-1])
    return np.array(X_test), unit_numbers, last_cycles


# ── Ana eğitim fonksiyonu ─────────────────────────────────────────────────

def train(data_dir: str, log_callback=None) -> dict:
    """
    Notebook pipeline'ını birebir çalıştırır.
    log_callback(msg: str) → canlı epoch logları için opsiyonel.
    Döndürür: metrikler + tahminler dict'i
    """
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, Callback

    import os, time
    t0 = time.time()

    # ── Veri yükle ────────────────────────────────────────────────────────
    train = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    # ── RUL hesapla (notebook Hücre 3) ────────────────────────────────────
    max_cycle = train.groupby("unit_number")["time_cycle"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    train = train.merge(max_cycle, on="unit_number", how="left")
    train["RUL"]        = train["max_cycle"] - train["time_cycle"]
    train["RUL_capped"] = train["RUL"].clip(upper=RUL_CAP)
    train = train.drop(columns=["max_cycle"])

    # ── Sabit kolonları çıkar ─────────────────────────────────────────────
    constant_cols = [col for col in train.columns if train[col].nunique() <= 1]
    train = train.drop(columns=constant_cols)
    test  = test.drop(columns=[c for c in constant_cols if c in test.columns])

    # ── Feature kolonları (notebook Hücre 3 çıktısı: 18 özellik) ─────────
    feature_cols = [
        col for col in train.columns
        if col not in ["unit_number", "RUL", "RUL_capped"]
    ]

    # ── Motor bazlı train/val split 80/20 (notebook Hücre 4) ─────────────
    unique_units              = train["unit_number"].unique()
    train_units, val_units    = train_test_split(unique_units, test_size=0.2, random_state=42)
    train_data = train[train["unit_number"].isin(train_units)].copy()
    val_data   = train[train["unit_number"].isin(val_units)].copy()

    # ── MinMaxScaler (notebook Hücre 4) ───────────────────────────────────
    scaler = MinMaxScaler()
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    val_data[feature_cols]   = scaler.transform(val_data[feature_cols])
    test[feature_cols]       = scaler.transform(test[feature_cols])

    # ── Sequence oluştur ──────────────────────────────────────────────────
    X_train_lstm, y_train_lstm = create_sequences(train_data, feature_cols, "RUL_capped", WINDOW_SIZE)
    X_val_lstm,   y_val_lstm   = create_sequences(val_data,   feature_cols, "RUL_capped", WINDOW_SIZE)
    X_test_lstm, test_units_out, test_last_cycles = create_test_sequences(test, feature_cols, WINDOW_SIZE)

    # ── Model (notebook Hücre 5 — birebir) ────────────────────────────────
    n_features  = X_train_lstm.shape[2]
    epoch_log   = []

    class EpochLog(Callback):
        def on_epoch_end(self, epoch, logs=None):
            entry = {
                "epoch":    epoch + 1,
                "loss":     round(float(logs.get("loss", 0)), 4),
                "val_loss": round(float(logs.get("val_loss", 0)), 4),
                "mae":      round(float(logs.get("mae", 0)), 4),
            }
            epoch_log.append(entry)
            if log_callback:
                log_callback(entry)

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=64, return_sequences=True,  input_shape=(WINDOW_SIZE, n_features)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=32, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(16, activation="relu"))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    lstm_model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=50, batch_size=64,
        callbacks=[early_stop, EpochLog()],
        verbose=0,
    )

    # ── Validation metrikleri ─────────────────────────────────────────────
    y_pred_val = lstm_model.predict(X_val_lstm, verbose=0).flatten()
    mae_v  = float(mean_absolute_error(y_val_lstm, y_pred_val))
    rmse_v = float(np.sqrt(mean_squared_error(y_val_lstm, y_pred_val)))
    r2_v   = float(r2_score(y_val_lstm, y_pred_val))

    # ── Test tahminleri ───────────────────────────────────────────────────
    test_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    # Risk seviyesi (notebook Hücre 6)
    def risk_from_rul(rul):
        if rul <= 30:  return "Yüksek Risk"
        if rul <= 60:  return "Orta Risk"
        return "Düşük Risk"

    def maintenance_from_rul(rul):
        if rul <= 30:  return "Acil bakım planına alınmalı"
        if rul <= 60:  return "Yakından izlenmeli ve bakım planına dahil edilmeli"
        return "Normal izleme devam etmeli"

    results_df = pd.DataFrame({
        "unit_number":        test_units_out,
        "time_cycle":         test_last_cycles,
        "predicted_RUL_LSTM": test_preds,
    })
    results_df["risk_level"]               = results_df["predicted_RUL_LSTM"].apply(risk_from_rul)
    results_df["maintenance_recommendation"]= results_df["predicted_RUL_LSTM"].apply(maintenance_from_rul)

    return {
        "name":            "LSTM",
        "type":            "regression",
        "mae":             round(mae_v,  3),
        "rmse":            round(rmse_v, 3),
        "r2":              round(r2_v,   4),
        "duration_s":      round(time.time() - t0, 1),
        "epochs":          len(epoch_log),
        "epoch_log":       epoch_log,
        "predictions":     test_preds.tolist(),
        "val_predictions": y_pred_val.tolist(),
        "val_actual":      y_val_lstm.tolist(),
        "actual":          rul_df["RUL"].values.clip(0, RUL_CAP).tolist(),
        "results_df":      results_df.to_dict(orient="records"),
        "model_obj":       lstm_model,
    }