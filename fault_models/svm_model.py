"""
SVM Arıza Tespiti — RF pipeline ile aynı yapı
RUL <= 30 cycle → arıza (1), üzeri → normal (0)
Çıktı: F1, Precision, Recall, AUC-ROC, Confusion Matrix, probabilities
"""
import warnings
warnings.filterwarnings("ignore")

import os, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

THRESHOLD = 30

COLUMN_NAMES = [
    "unit","cycle","op1","op2","op3",
    "s1","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","s12","s13",
    "s14","s15","s16","s17","s18","s19","s20","s21",
]

USEFUL_SENSORS = [
    "s1","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","s12","s13",
    "s14","s15","s16","s17","s18","s19","s20","s21",
]


def _ozellik_cikar(df, sensors, window=30, etiket=True):
    kayitlar = []
    for unit_id, group in df.groupby("unit"):
        group    = group.reset_index(drop=True)
        satirlar = range(len(group)) if etiket else [len(group) - 1]
        for i in satirlar:
            pen   = group[sensors].iloc[max(0, i - window + 1):i + 1]
            satir = {"unit": unit_id}
            for s in sensors:
                v = pen[s].values
                satir[f"{s}_son"]   = v[-1]
                satir[f"{s}_ort"]   = np.mean(v)
                satir[f"{s}_std"]   = np.std(v)
                satir[f"{s}_min"]   = np.min(v)
                satir[f"{s}_max"]   = np.max(v)
                satir[f"{s}_trend"] = v[-1] - v[0]
            if etiket:
                satir["ariza"] = group["ariza"].iloc[i]
            kayitlar.append(satir)
    return pd.DataFrame(kayitlar)


def train(data_dir: str, log_callback=None) -> dict:
    def log(msg):
        if log_callback:
            log_callback(msg)

    t0 = time.time()

    # ── Veri yükle ──────────────────────────────────────────────────────
    train_df = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    log(f"Train: {train_df.shape}  Test: {test_df.shape}")

    # ── RUL ve arıza etiketi ────────────────────────────────────────────
    mc = train_df.groupby("unit")["cycle"].max().reset_index()
    mc.columns = ["unit", "max_cycle"]
    train_df   = train_df.merge(mc, on="unit")
    train_df["RUL"]   = train_df["max_cycle"] - train_df["cycle"]
    train_df["ariza"] = (train_df["RUL"] <= THRESHOLD).astype(int)
    train_df.drop("max_cycle", axis=1, inplace=True)

    y_test = (rul_df["RUL"].values <= THRESHOLD).astype(int)
    log(f"Arıza oranı (train): {train_df['ariza'].mean()*100:.1f}%")

    # ── Özellik çıkarımı ────────────────────────────────────────────────
    log("Özellik çıkarılıyor...")
    train_feat = _ozellik_cikar(train_df, USEFUL_SENSORS, window=30, etiket=True)
    test_feat  = _ozellik_cikar(test_df,  USEFUL_SENSORS, window=30, etiket=False)

    feature_cols = [c for c in train_feat.columns if c not in ["unit", "ariza"]]
    X_train_raw  = train_feat[feature_cols].values
    y_train      = train_feat["ariza"].values
    X_test_raw   = test_feat[feature_cols].values
    log(f"X_train: {X_train_raw.shape}  X_test: {X_test_raw.shape}")

    # ── StandardScaler ──────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # ── SVM — RF gibi balanced + probability=True ────────────────────────
    log("SVM eğitiliyor (C=1, rbf, balanced)...")
    model = SVC(
        kernel="rbf",
        C=1,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_test, y_prob))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    log(f"SVM — F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  AUC={auc:.4f}")

    return {
        "name":             "SVM (RBF)",
        "type":             "classification",
        "f1":               round(f1,   4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "auc":              round(auc,  4),
        "confusion_matrix": cm,
        "probabilities":    y_prob.tolist(),
        "predictions":      y_pred.tolist(),
        "actual":           y_test.tolist(),
        "best_params":      {"C": 1, "kernel": "rbf", "class_weight": "balanced"},
        "duration_s":       round(time.time() - t0, 1),
        "model_obj":        model,
        "scaler_obj":       scaler,
    }