"""
Random Forest Arıza Tespiti — RF_Basit.ipynb (Deney 4: en iyi, leaf=5) pipeline
train_FD001.txt / test_FD001.txt / RUL_FD001.txt
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


COL_NAMES = [
    "unit", "cycle", "op1", "op2", "op3",
    "s1","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","s12","s13",
    "s14","s15","s16","s17","s18","s19","s20","s21",
]
USEFUL_SENSORS = [
    "s2","s3","s4","s7","s8","s9",
    "s11","s12","s13","s14","s15",
    "s17","s20","s21",
]
THRESHOLD = 30
WINDOW    = 30


# ── Özellik çıkarma (notebook Hücre 4 — ozellik_cikar) ───────────────────

def ozellik_cikar(df, sensors, window, etiket=True):
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
                satir[f"{s}_ort"]   = float(np.mean(v))
                satir[f"{s}_std"]   = float(np.std(v))
                satir[f"{s}_min"]   = float(np.min(v))
                satir[f"{s}_max"]   = float(np.max(v))
                satir[f"{s}_trend"] = float(v[-1] - v[0])
            if etiket:
                satir["ariza"] = int(group["ariza"].iloc[i])
            kayitlar.append(satir)
    return pd.DataFrame(kayitlar)


# ── Ana eğitim fonksiyonu ─────────────────────────────────────────────────

def train(data_dir: str, log_callback=None) -> dict:
    """
    RF_Basit.ipynb — Deney 4 (en iyi: n=300, depth=10, leaf=5) pipeline birebir.
    BEST_T = 0.35 threshold ile final tahmin.
    """
    import os
    t0 = time.time()

    # ── Veri yükle (notebook Hücre 2) ─────────────────────────────────────
    train_df = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COL_NAMES
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COL_NAMES
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    # ── Etiket oluştur (notebook Hücre 3) ─────────────────────────────────
    mc = train_df.groupby("unit")["cycle"].max().reset_index()
    mc.columns = ["unit", "max_cycle"]
    train_df   = train_df.merge(mc, on="unit")
    train_df["RUL"]   = train_df["max_cycle"] - train_df["cycle"]
    train_df["ariza"] = (train_df["RUL"] <= THRESHOLD).astype(int)
    train_df.drop("max_cycle", axis=1, inplace=True)

    y_test = (rul_df["RUL"].values <= THRESHOLD).astype(int)

    if log_callback:
        log_callback(f"Arıza oranı: {train_df['ariza'].mean()*100:.1f}%")

    # ── Özellik çıkar (notebook Hücre 4) ──────────────────────────────────
    if log_callback:
        log_callback("Özellik çıkarılıyor...")

    train_feat = ozellik_cikar(train_df, USEFUL_SENSORS, WINDOW, etiket=True)
    test_feat  = ozellik_cikar(test_df,  USEFUL_SENSORS, WINDOW, etiket=False)

    feature_cols = [c for c in train_feat.columns if c not in ["unit", "ariza"]]
    X_train_raw  = train_feat[feature_cols].values
    y_train      = train_feat["ariza"].values
    X_test_raw   = test_feat[feature_cols].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    if log_callback:
        log_callback(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ── Deney 4 — en iyi model (notebook Hücre 9) ─────────────────────────
    model4 = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    model4.fit(X_train, y_train)
    prob4 = model4.predict_proba(X_test)[:, 1]

    # ── Threshold analizi — BEST_T = 0.35 (notebook Hücre 13) ─────────────
    BEST_T        = 0.35
    y_pred_final  = (prob4 >= BEST_T).astype(int)

    f1   = float(f1_score(y_test,   y_pred_final, zero_division=0))
    prec = float(precision_score(y_test, y_pred_final, zero_division=0))
    rec  = float(recall_score(y_test,    y_pred_final, zero_division=0))
    auc  = float(roc_auc_score(y_test, prob4))
    cm   = confusion_matrix(y_test, y_pred_final).tolist()

    if log_callback:
        log_callback(f"F1={f1:.4f} AUC={auc:.4f} threshold={BEST_T}")

    # ── Feature importance ─────────────────────────────────────────────────
    feat_imp = dict(zip(feature_cols, model4.feature_importances_.tolist()))
    top10    = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10])

    return {
        "name":               "Random Forest",
        "type":               "classification",
        "f1":                 round(f1,   4),
        "precision":          round(prec, 4),
        "recall":             round(rec,  4),
        "auc":                round(auc,  4),
        "threshold":          BEST_T,
        "confusion_matrix":   cm,
        "feature_importance": top10,
        "duration_s":         round(time.time() - t0, 1),
        "predictions":        y_pred_final.tolist(),
        "probabilities":      prob4.tolist(),
        "actual":             y_test.tolist(),
        "model_obj":          model4,
        "scaler_obj":         scaler,
        "feature_cols":       feature_cols,
    }