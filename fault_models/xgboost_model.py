"""
XGBoost RUL + Arıza Tespiti — XGBoost_arıza.ipynb pipeline birebir
train_FD001.txt / test_FD001.txt / RUL_FD001.txt
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


COLUMN_NAMES = [
    "unit_number", "time_cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]
RUL_CAP   = 125
THRESHOLD = 30


def train(data_dir: str, log_callback=None) -> dict:
    """
    XGBoost_arıza.ipynb pipeline birebir.
    RUL regressor + failure_label classifier — ikisi birden.
    """
    from xgboost import XGBRegressor, XGBClassifier
    import os
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

    # ── Sütun isimlerini eşle (notebook — column_names listesi) ───────────
    if train_df.shape[1] == 26:
        train_df.columns = COLUMN_NAMES
    if test_df.shape[1] == 26:
        test_df.columns = COLUMN_NAMES

    # ── RUL hesapla (notebook) ─────────────────────────────────────────────
    max_cycle = train_df.groupby("unit_number")["time_cycle"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    train_df = train_df.merge(max_cycle, on="unit_number", how="left")
    train_df["RUL"]         = train_df["max_cycle"] - train_df["time_cycle"]
    train_df["failure_label"] = np.where(train_df["RUL"] <= THRESHOLD, 1, 0)
    train_df["RUL_capped"]  = train_df["RUL"].clip(upper=RUL_CAP)
    train_df = train_df.drop(columns=["max_cycle"])

    if log_callback:
        log_callback(f"Arıza (1): {train_df['failure_label'].sum():,}  Normal (0): {(train_df['failure_label']==0).sum():,}")

    # ── Sabit kolonları çıkar ─────────────────────────────────────────────
    constant_cols = [col for col in train_df.columns if train_df[col].nunique() <= 1]
    train_df = train_df.drop(columns=constant_cols)
    test_df  = test_df.drop(columns=[c for c in constant_cols if c in test_df.columns])

    # ── Motor bazlı train/val split 80/20 (notebook) ──────────────────────
    drop_cols_rul = ["RUL", "RUL_capped", "failure_label", "unit_number"]
    unique_units  = train_df["unit_number"].unique()
    train_units, val_units = train_test_split(unique_units, test_size=0.2, random_state=42)
    train_data = train_df[train_df["unit_number"].isin(train_units)]
    val_data   = train_df[train_df["unit_number"].isin(val_units)]

    X_train_rul = train_data.drop(columns=[c for c in drop_cols_rul if c in train_data.columns])
    y_train_rul = train_data["RUL_capped"]
    X_val_rul   = val_data.drop(columns=[c for c in drop_cols_rul if c in val_data.columns])
    y_val_rul   = val_data["RUL_capped"]

    drop_cols_cls = ["RUL", "RUL_capped", "failure_label", "unit_number"]
    X_train_cls = train_data.drop(columns=[c for c in drop_cols_cls if c in train_data.columns])
    y_train_cls = train_data["failure_label"]
    X_val_cls   = val_data.drop(columns=[c for c in drop_cols_cls if c in val_data.columns])
    y_val_cls   = val_data["failure_label"]

    if log_callback:
        log_callback(f"X_train: {X_train_rul.shape}  X_val: {X_val_rul.shape}")

    # ── XGBoost RUL Regressor (notebook) ──────────────────────────────────
    xgb_rul = XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, verbosity=0,
    )
    xgb_rul.fit(X_train_rul, y_train_rul)
    y_pred_rul = xgb_rul.predict(X_val_rul)

    mae_v  = float(mean_absolute_error(y_val_rul, y_pred_rul))
    rmse_v = float(np.sqrt(mean_squared_error(y_val_rul, y_pred_rul)))
    r2_v   = float(r2_score(y_val_rul, y_pred_rul))

    if log_callback:
        log_callback(f"XGBoost RUL — MAE={mae_v:.2f} RMSE={rmse_v:.2f} R2={r2_v:.4f}")

    # ── XGBoost Failure Classifier (notebook) ─────────────────────────────
    xgb_failure = XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_failure.fit(X_train_cls, y_train_cls)
    y_pred_cls = xgb_failure.predict(X_val_cls)
    y_prob_cls = xgb_failure.predict_proba(X_val_cls)[:, 1]

    acc  = float(accuracy_score(y_val_cls,   y_pred_cls))
    prec = float(precision_score(y_val_cls,  y_pred_cls, zero_division=0))
    rec  = float(recall_score(y_val_cls,     y_pred_cls, zero_division=0))
    f1   = float(f1_score(y_val_cls,         y_pred_cls, zero_division=0))
    auc  = float(roc_auc_score(y_val_cls,    y_prob_cls))
    cm   = confusion_matrix(y_val_cls,       y_pred_cls).tolist()

    if log_callback:
        log_callback(f"XGBoost Cls — F1={f1:.4f} AUC={auc:.4f} Acc={acc:.4f}")

    # ── Test seti — son satır (notebook: test.groupby.tail(1)) ────────────
    test_last = test_df.groupby("unit_number").tail(1).copy()

    X_test_rul = test_last.copy().drop(columns=["unit_number"], errors="ignore")
    X_test_rul = X_test_rul[X_train_rul.columns]

    test_last["predicted_RUL"]       = xgb_rul.predict(X_test_rul)

    X_test_cls = test_last.copy().drop(columns=["unit_number", "predicted_RUL"], errors="ignore")
    X_test_cls = X_test_cls[X_train_cls.columns]

    test_last["failure_probability"] = xgb_failure.predict_proba(X_test_cls)[:, 1]

    # Risk seviyesi (notebook)
    def risk_level(prob, rul):
        if prob >= 0.70 or rul <= 30:  return "Yüksek Risk"
        if prob >= 0.40 or rul <= 60:  return "Orta Risk"
        return "Düşük Risk"

    def maintenance_recommendation(risk):
        if risk == "Yüksek Risk": return "Acil bakım planına alınmalı"
        if risk == "Orta Risk":   return "Yakından izlenmeli ve bakım planına dahil edilmeli"
        return "Normal izleme devam etmeli"

    test_last["risk_level"] = test_last.apply(
        lambda row: risk_level(row["failure_probability"], row["predicted_RUL"]), axis=1
    )
    test_last["maintenance_recommendation"] = test_last["risk_level"].apply(maintenance_recommendation)
    test_last["failure_probability_percent"] = test_last["failure_probability"] * 100

    # Test metrikleri (RUL_FD001 ile karşılaştır)
    y_test_real = rul_df["RUL"].values.clip(0, RUL_CAP)
    test_preds  = test_last["predicted_RUL"].values
    rmse_test   = float(np.sqrt(mean_squared_error(y_test_real, test_preds)))
    mae_test    = float(mean_absolute_error(y_test_real, test_preds))
    r2_test     = float(r2_score(y_test_real, test_preds))

    y_test_ariza = (rul_df["RUL"].values <= THRESHOLD).astype(int)
    test_probs   = test_last["failure_probability"].values
    auc_test     = float(roc_auc_score(y_test_ariza, test_probs))

    return {
        "name":          "XGBoost",
        "type":          "both",
        # Regression
        "mae":           round(mae_test,  3),
        "rmse":          round(rmse_test, 3),
        "r2":            round(r2_test,   4),
        "val_mae":       round(mae_v,  3),
        "val_rmse":      round(rmse_v, 3),
        "val_r2":        round(r2_v,   4),
        # Classification
        "accuracy":      round(acc,  4),
        "precision":     round(prec, 4),
        "recall":        round(rec,  4),
        "f1":            round(f1,   4),
        "auc":           round(auc,  4),
        "auc_test":      round(auc_test, 4),
        "confusion_matrix": cm,
        "duration_s":    round(time.time() - t0, 1),
        "predictions":   test_preds.tolist(),
        "probabilities": test_probs.tolist(),
        "actual":        y_test_real.tolist(),
        "results_df":    test_last[["unit_number","time_cycle","predicted_RUL",
                                    "failure_probability_percent","risk_level",
                                    "maintenance_recommendation"]].to_dict(orient="records"),
        "model_obj":     {"xgb_rul": xgb_rul, "xgb_failure": xgb_failure},
    }