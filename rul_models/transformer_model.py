"""
TST (Time Series Transformer) RUL Tahmini
Sadece tek ve en iyi parametre kombinasyonu çalışır.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

COLUMN_NAMES = [
    "unit", "cycle",
    "op1", "op2", "op3",
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13",
    "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

USEFUL_SENSORS = [
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13",
    "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
]

MAX_RUL = 125


# ─────────────────────────────────────────────────────────────
# Sequence oluşturma
# ─────────────────────────────────────────────────────────────

def make_sequences(df, sensors, window):
    X_list, y_list = [], []

    for _, group in df.groupby("unit"):
        data = group[sensors].values
        rul = group["RUL"].values

        for i in range(len(data) - window):
            X_list.append(data[i:i + window].T)
            y_list.append(rul[i + window])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32)
    )


def make_test_sequences(df, sensors, window):
    X_list = []

    for _, group in df.groupby("unit"):
        data = group[sensors].values

        if len(data) >= window:
            X_list.append(data[-window:].T)

        else:
            pad = np.zeros((window - len(data), len(sensors)))
            padded = np.vstack([pad, data])
            X_list.append(padded.T)

    return np.array(X_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Metrikler
# ─────────────────────────────────────────────────────────────

def calculate_metrics(y_true, preds):
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))

    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────
# Ana eğitim fonksiyonu
# ─────────────────────────────────────────────────────────────

def train(data_dir: str, log_callback=None):

    import torch

    from tsai.all import (
        get_splits,
        TSDatasets,
        TSRegression,
        Learner,
        MSELossFlat,
        rmse,
        mae,
        TST,
    )

    start_time = time.time()

    # ─────────────────────────────────────────────────────────
    # Veri yükleme
    # ─────────────────────────────────────────────────────────

    train_df = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    test_df = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    rul_df = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    # ─────────────────────────────────────────────────────────
    # RUL oluşturma
    # ─────────────────────────────────────────────────────────

    def add_rul(df):
        max_cycle = df.groupby("unit")["cycle"].max().reset_index()
        max_cycle.columns = ["unit", "max_cycle"]

        df = df.merge(max_cycle, on="unit")

        df["RUL"] = df["max_cycle"] - df["cycle"]

        df.drop("max_cycle", axis=1, inplace=True)

        return df

    train_df = add_rul(train_df)

    # ─────────────────────────────────────────────────────────
    # Normalizasyon
    # ─────────────────────────────────────────────────────────

    scaler = StandardScaler()

    train_df[USEFUL_SENSORS] = scaler.fit_transform(
        train_df[USEFUL_SENSORS]
    )

    test_df[USEFUL_SENSORS] = scaler.transform(
        test_df[USEFUL_SENSORS]
    )

    train_df["RUL"] = (
        train_df["RUL"].clip(upper=MAX_RUL) / MAX_RUL
    )

    # ─────────────────────────────────────────────────────────
    # EN İYİ PARAMETRELER
    # ─────────────────────────────────────────────────────────

    WINDOW = 50
    D_MODEL = 128
    N_HEADS = 8
    D_FF = 512
    DROPOUT = 0.2
    N_LAYERS = 3
    EPOCHS = 60
    LR = 1e-3
    BATCH_SIZE = 64

    # ─────────────────────────────────────────────────────────
    # Sequence oluştur
    # ─────────────────────────────────────────────────────────

    X_train, y_train = make_sequences(
        train_df,
        USEFUL_SENSORS,
        WINDOW
    )

    X_test = make_test_sequences(
        test_df,
        USEFUL_SENSORS,
        WINDOW
    )

    y_test = (
        rul_df["RUL"].values.astype(np.float32) / MAX_RUL
    )

    # ─────────────────────────────────────────────────────────
    # Train / Validation split
    # ─────────────────────────────────────────────────────────

    splits = get_splits(
        y_train,
        valid_size=0.2,
        shuffle=True,
        random_state=42
    )

    tfms = [None, [TSRegression()]]

    dsets = TSDatasets(
        X_train,
        y_train,
        tfms=tfms,
        splits=splits
    )

    dls = dsets.dataloaders(
        bs=BATCH_SIZE,
        num_workers=0
    )

    # ─────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────

    model = TST(
        c_in=len(USEFUL_SENSORS),
        c_out=1,
        seq_len=WINDOW,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
    )

    learner = Learner(
        dls,
        model,
        loss_func=MSELossFlat(),
        metrics=[rmse, mae]
    )

    # ─────────────────────────────────────────────────────────
    # Eğitim
    # ─────────────────────────────────────────────────────────

    learner.fit_one_cycle(EPOCHS, lr_max=LR)

    # ─────────────────────────────────────────────────────────
    # Loss eğrileri
    # ─────────────────────────────────────────────────────────

    train_curve = []
    valid_curve = []

    try:
        for vals in learner.recorder.values:
            if len(vals) >= 2:
                train_curve.append(float(vals[0]))
                valid_curve.append(float(vals[1]))
    except:
        pass

    # ─────────────────────────────────────────────────────────
    # Tahmin
    # ─────────────────────────────────────────────────────────

    model.eval()

    with torch.no_grad():

        X_test_tensor = torch.tensor(X_test).to(
            learner.dls.device
        )

        preds = (
            model(X_test_tensor)
            .cpu()
            .numpy()
            .squeeze()
        )

    preds = preds * MAX_RUL
    y_true = y_test * MAX_RUL

    # ─────────────────────────────────────────────────────────
    # Metrikler
    # ─────────────────────────────────────────────────────────

    rmse_val, mae_val, r2_val = calculate_metrics(
        y_true,
        preds
    )

    if log_callback:
        log_callback(
            f"RMSE={rmse_val:.2f} | "
            f"MAE={mae_val:.2f} | "
            f"R2={r2_val:.4f}"
        )

    # ─────────────────────────────────────────────────────────
    # Sonuç
    # ─────────────────────────────────────────────────────────

    return {
        "name": "TST (tsai)",
        "type": "regression",

        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        "r2": round(r2_val, 4),

        "epochs": EPOCHS,

        "duration_s": round(time.time() - start_time, 1),

        "train_curve": train_curve,
        "valid_curve": valid_curve,

        "predictions": preds.tolist(),
        "actual": y_true.tolist(),

        "params": {
            "window": WINDOW,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "n_layers": N_LAYERS,
            "epochs": EPOCHS,
            "lr": LR,
        },

        "model_obj": model,
    }