"""
GRU RUL Tahmini — gru_modelleme.ipynb (GRU v3, en iyi model) pipeline
train_FD001.txt / test_FD001.txt / RUL_FD001.txt üzerinde çalışır.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


COLUMN_NAMES = [
    "unit_number", "time_cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]
RUL_CAP     = 125
WINDOW_SIZE = 30


# ── Sequence oluşturucular (notebook Hücre 47) ────────────────────────────

def create_sequences(df, feature_cols, window_size=30):
    X, y = [], []
    for unit in df["unit_number"].unique():
        unit_df  = df[df["unit_number"] == unit].sort_values("time_cycle")
        features = unit_df[feature_cols].values
        target   = unit_df["RUL"].values
        for i in range(len(unit_df) - window_size):
            X.append(features[i:i + window_size])
            y.append(target[i + window_size])
    return np.array(X), np.array(y)


def create_test_sequences(test_data, feature_cols, window_size=30):
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
    gru_modelleme.ipynb GRU v3 pipeline — birebir.
    """
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
    from keras.optimizers import Adam
    import os, time
    t0 = time.time()

    # ── Veri yükle ────────────────────────────────────────────────────────
    train_df = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    # ── Her motorun maksimum cycle sayısını bul, RUL hesapla (notebook Hücre 13) ──
    max_cycles = train_df.groupby("unit_number")["time_cycle"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="unit_number")
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_cycle"]
    train_df.drop(columns=["max_cycle"], inplace=True)

    # RUL_CAP uygula (notebook Hücre 34)
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)

    # ── Sabit kolonları çıkar (notebook Hücre 14) ─────────────────────────
    constant_cols = [col for col in train_df.columns if train_df[col].nunique() <= 1]
    train_df.drop(columns=constant_cols, inplace=True)
    test_df.drop(columns=[c for c in constant_cols if c in test_df.columns], inplace=True)

    # ── Feature kolonları (notebook Hücre 36 — sabit + op ayarları hariç) ─
    drop_cols    = ["unit_number", "time_cycle", "RUL"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    # ── Motor bazlı train/test split 80/20 (notebook Hücre 37) ────────────
    from sklearn.model_selection import train_test_split
    units       = train_df["unit_number"].unique()
    train_units, test_units_split = train_test_split(units, test_size=0.2, random_state=42)
    train_split = train_df[train_df["unit_number"].isin(train_units)].copy()
    val_split   = train_df[train_df["unit_number"].isin(test_units_split)].copy()

    # ── StandardScaler — sadece X (notebook Hücre 38) ────────────────────
    scaler = StandardScaler()
    train_split[feature_cols] = scaler.fit_transform(train_split[feature_cols])
    val_split[feature_cols]   = scaler.transform(val_split[feature_cols])
    test_df[feature_cols]     = scaler.transform(test_df[feature_cols])

    # ── Sequence oluştur (notebook Hücre 47-48, normalize /125) ──────────
    X_train_seq, y_train_seq = create_sequences(train_split, feature_cols, WINDOW_SIZE)
    X_test_seq,  y_test_seq  = create_sequences(val_split,   feature_cols, WINDOW_SIZE)
    y_train_seq = y_train_seq / RUL_CAP
    y_test_seq  = y_test_seq  / RUL_CAP

    # Test seti için son sequence
    X_final_test, test_units_out, test_last_cycles = create_test_sequences(test_df, feature_cols, WINDOW_SIZE)

    # ── EpochLog callback ─────────────────────────────────────────────────
    epoch_log = []

    class EpochLog(Callback):
        def on_epoch_end(self, epoch, logs=None):
            entry = {
                "epoch":    epoch + 1,
                "loss":     round(float(logs.get("loss", 0)), 4),
                "val_loss": round(float(logs.get("val_loss", 0)), 4),
            }
            epoch_log.append(entry)
            if log_callback:
                log_callback(entry)

    # ── Model — GRU v3 (notebook Hücre 49 + Hücre 32) ────────────────────
    n_features = len(feature_cols)
    model_gru = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),

        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model_gru.compile(optimizer=Adam(learning_rate=0.001), loss="huber")

    # ── Callbacks (notebook Hücre 33) ─────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        EpochLog(),
    ]

    model_gru.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=100, batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    # ── Validation metrikleri ─────────────────────────────────────────────
    y_pred_scaled = model_gru.predict(X_test_seq, verbose=0).flatten()
    y_pred_val    = y_pred_scaled * RUL_CAP
    y_val_real    = y_test_seq   * RUL_CAP

    mae_v  = float(mean_absolute_error(y_val_real, y_pred_val))
    rmse_v = float(np.sqrt(mean_squared_error(y_val_real, y_pred_val)))
    r2_v   = float(r2_score(y_val_real, y_pred_val))

    # ── Test tahmini ──────────────────────────────────────────────────────
    test_preds_scaled = model_gru.predict(X_final_test, verbose=0).flatten()
    test_preds        = test_preds_scaled * RUL_CAP

    # Gerçek RUL (test seti için)
    y_test_real = rul_df["RUL"].values.clip(0, RUL_CAP)

    rmse_test = float(np.sqrt(mean_squared_error(y_test_real, test_preds)))
    mae_test  = float(mean_absolute_error(y_test_real, test_preds))
    r2_test   = float(r2_score(y_test_real, test_preds))

    return {
        "name":            "GRU",
        "type":            "regression",
        "mae":             round(mae_test,  3),
        "rmse":            round(rmse_test, 3),
        "r2":              round(r2_test,   4),
        "val_mae":         round(mae_v,  3),
        "val_rmse":        round(rmse_v, 3),
        "val_r2":          round(r2_v,   4),
        "duration_s":      round(time.time() - t0, 1),
        "epochs":          len(epoch_log),
        "epoch_log":       epoch_log,
        "predictions":     test_preds.tolist(),
        "val_predictions": y_pred_val.tolist(),
        "val_actual":      y_val_real.tolist(),
        "actual":          y_test_real.tolist(),
        "model_obj":       model_gru,
    }