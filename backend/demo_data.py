"""
HELIOS PHM — Demo Modu
Gerçek veri/model yoksa önceden hesaplanmış gerçekçi sonuçlar döner.
NASA C-MAPSS FD001 gerçek eğitim sonuçlarına yakın değerler.
"""
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# 100 test motoru için gerçekçi RUL tahminleri (LSTM benzeri)
_actual_rul = [
    112, 98, 69, 82, 91, 93, 91, 95, 111, 96,
    97,  124, 95, 107, 83, 84, 50, 28, 87, 16,
    57,  111, 113, 20, 145, 119, 60, 13, 55, 23,
    22,  63, 26, 0, 23, 19, 82, 154, 46, 117,
    81,  75, 30, 20, 66, 172, 77, 159, 61, 48,
    38,  32, 86, 11, 93, 87, 20, 53, 45, 6,
    11,  35, 46, 33, 48, 18, 30, 48, 10, 5,
    55,  1, 50, 49, 26, 56, 43, 46, 26, 49,
    28,  1, 20, 58, 3, 52, 25, 36, 29, 37,
    45,  28, 9, 23, 45, 40, 46, 45, 6, 9,
]

def _add_noise(vals, sigma=4.0):
    return [max(0, v + float(np.random.normal(0, sigma))) for v in vals]

def _clamp(v, lo=0, hi=125):
    return max(lo, min(hi, v))

def load_demo_results():
    """
    Tüm modeller için demo RESULTS dict'i döner.
    app.py başlangıcında RESULTS boşsa bu çağrılır.
    """
    actual = [_clamp(v) for v in _actual_rul]
    n = len(actual)

    # ── LSTM ──────────────────────────────────────────────────────────────
    lstm_preds = _add_noise(actual, sigma=6)
    lstm_val_actual = [_clamp(float(np.random.normal(60, 30))) for _ in range(400)]
    lstm_val_preds  = _add_noise(lstm_val_actual, sigma=7)
    lstm_epoch_log  = [
        {"epoch": e+1,
         "loss":     round(0.08 * (0.92 ** e) + 0.012, 4),
         "val_loss": round(0.09 * (0.91 ** e) + 0.015, 4),
         "mae":      round(12.0 * (0.93 ** e) + 4.0,   4)}
        for e in range(32)
    ]
    lstm_r2 = 0.7821
    lstm_rmse = 24.31
    lstm_mae  = 17.42

    lstm = {
        "name": "LSTM", "type": "regression",
        "mae": lstm_mae, "rmse": lstm_rmse, "r2": lstm_r2,
        "val_mae": 18.12, "val_rmse": 25.67, "val_r2": 0.7654,
        "duration_s": 48.3,
        "epochs": 32,
        "epoch_log": lstm_epoch_log,
        "predictions": lstm_preds,
        "val_predictions": lstm_val_preds,
        "val_actual": lstm_val_actual,
        "actual": actual,
        "results_df": [],
    }

    # ── GRU ───────────────────────────────────────────────────────────────
    gru_preds = _add_noise(actual, sigma=5.5)
    gru_epoch_log = [
        {"epoch": e+1,
         "loss":     round(0.075 * (0.91 ** e) + 0.011, 4),
         "val_loss": round(0.085 * (0.90 ** e) + 0.014, 4)}
        for e in range(38)
    ]
    gru_val_actual = [_clamp(float(np.random.normal(60, 28))) for _ in range(400)]
    gru_val_preds  = _add_noise(gru_val_actual, sigma=6)

    gru = {
        "name": "GRU", "type": "regression",
        "mae": 16.89, "rmse": 23.14, "r2": 0.8012,
        "val_mae": 17.45, "val_rmse": 24.21, "val_r2": 0.7891,
        "duration_s": 41.7,
        "epochs": 38,
        "epoch_log": gru_epoch_log,
        "predictions": gru_preds,
        "val_predictions": gru_val_preds,
        "val_actual": gru_val_actual,
        "actual": actual,
    }

    # ── TST (Transformer) ─────────────────────────────────────────────────
    tst_preds = _add_noise(actual, sigma=7)
    tst_train_curve = [round(0.082 * (0.93 ** e) + 0.013, 4) for e in range(60)]
    tst_valid_curve = [round(0.091 * (0.92 ** e) + 0.016, 4) for e in range(60)]

    tst = {
        "name": "TST (tsai)", "type": "regression",
        "mae": 18.03, "rmse": 25.87, "r2": 0.7543,
        "duration_s": 124.6,
        "epochs": 60,
        "train_curve": tst_train_curve,
        "valid_curve": tst_valid_curve,
        "predictions": tst_preds,
        "actual": actual,
        "params": {
            "window": 50, "d_model": 128, "n_heads": 8,
            "d_ff": 512, "dropout": 0.2, "n_layers": 3,
            "epochs": 60, "lr": 0.001,
        },
    }

    # ── Random Forest ──────────────────────────────────────────────────────
    y_test_binary = [1 if v <= 30 else 0 for v in actual]
    rf_probs = [
        min(1.0, max(0.0, (1 - v/125) * 0.85 + float(np.random.normal(0, 0.06))))
        for v in actual
    ]
    rf_preds = [1 if p >= 0.35 else 0 for p in rf_probs]

    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
    )
    rf_f1   = round(float(f1_score(y_test_binary, rf_preds, zero_division=0)), 4)
    rf_prec = round(float(precision_score(y_test_binary, rf_preds, zero_division=0)), 4)
    rf_rec  = round(float(recall_score(y_test_binary, rf_preds, zero_division=0)), 4)
    rf_auc  = round(float(roc_auc_score(y_test_binary, rf_probs)), 4)
    rf_cm   = confusion_matrix(y_test_binary, rf_preds).tolist()

    rf = {
        "name": "Random Forest", "type": "classification",
        "f1": rf_f1, "precision": rf_prec, "recall": rf_rec, "auc": rf_auc,
        "threshold": 0.35,
        "confusion_matrix": rf_cm,
        "feature_importance": {
            "s12_trend": 0.089, "s11_son": 0.071, "s4_ort": 0.063,
            "s14_std": 0.058, "s9_trend": 0.052, "s7_max": 0.048,
            "s2_ort": 0.044, "s3_son": 0.041, "s20_trend": 0.038, "s21_std": 0.034,
        },
        "duration_s": 18.4,
        "predictions": rf_preds,
        "probabilities": rf_probs,
        "actual": y_test_binary,
    }

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_probs = [
        min(1.0, max(0.0, (1 - v/125) * 0.88 + float(np.random.normal(0, 0.05))))
        for v in actual
    ]
    xgb_preds_rul = _add_noise(actual, sigma=5)
    xgb_preds_cls = [1 if p >= 0.5 else 0 for p in xgb_probs]
    xgb_cm  = confusion_matrix(y_test_binary, xgb_preds_cls).tolist()
    xgb_f1  = round(float(f1_score(y_test_binary, xgb_preds_cls, zero_division=0)), 4)
    xgb_prec= round(float(precision_score(y_test_binary, xgb_preds_cls, zero_division=0)), 4)
    xgb_rec = round(float(recall_score(y_test_binary, xgb_preds_cls, zero_division=0)), 4)
    xgb_auc = round(float(roc_auc_score(y_test_binary, xgb_probs)), 4)

    def risk_level(prob, rul):
        if prob >= 0.70 or rul <= 30: return "Yüksek Risk"
        if prob >= 0.40 or rul <= 60: return "Orta Risk"
        return "Düşük Risk"

    def maint_rec(risk):
        if risk == "Yüksek Risk": return "Acil bakım planına alınmalı"
        if risk == "Orta Risk":   return "Yakından izlenmeli ve bakım planına dahil edilmeli"
        return "Normal izleme devam etmeli"

    results_df = []
    for i, (rul_v, prob_v) in enumerate(zip(xgb_preds_rul, xgb_probs)):
        rul_v = max(0, rul_v)
        risk  = risk_level(prob_v, rul_v)
        results_df.append({
            "unit_number": i + 1,
            "time_cycle": 200 + i,
            "predicted_RUL": round(rul_v, 1),
            "failure_probability_percent": round(prob_v * 100, 2),
            "risk_level": risk,
            "maintenance_recommendation": maint_rec(risk),
        })

    xgb = {
        "name": "XGBoost", "type": "both",
        "mae": 15.21, "rmse": 21.87, "r2": 0.8234,
        "val_mae": 16.03, "val_rmse": 22.45, "val_r2": 0.8101,
        "accuracy": 0.8700, "precision": xgb_prec, "recall": xgb_rec,
        "f1": xgb_f1, "auc": xgb_auc, "auc_test": round(xgb_auc - 0.01, 4),
        "confusion_matrix": xgb_cm,
        "duration_s": 22.8,
        "predictions": xgb_preds_rul,
        "probabilities": xgb_probs,
        "actual": actual,
        "results_df": results_df,
    }

    # ── SVM ───────────────────────────────────────────────────────────────
    svm_probs = [
        min(1.0, max(0.0, (1 - v/125) * 0.80 + float(np.random.normal(0, 0.07))))
        for v in actual
    ]
    svm_preds = [1 if p >= 0.5 else 0 for p in svm_probs]
    svm_cm   = confusion_matrix(y_test_binary, svm_preds).tolist()
    svm_f1   = round(float(f1_score(y_test_binary, svm_preds, zero_division=0)), 4)
    svm_prec = round(float(precision_score(y_test_binary, svm_preds, zero_division=0)), 4)
    svm_rec  = round(float(recall_score(y_test_binary, svm_preds, zero_division=0)), 4)
    svm_auc  = round(float(roc_auc_score(y_test_binary, svm_probs)), 4)

    svm = {
        "name": "SVM (RBF)", "type": "classification",
        "f1": svm_f1, "precision": svm_prec, "recall": svm_rec, "auc": svm_auc,
        "confusion_matrix": svm_cm,
        "best_params": {"C": 1, "kernel": "rbf", "class_weight": "balanced"},
        "duration_s": 35.2,
        "predictions": svm_preds,
        "probabilities": svm_probs,
        "actual": y_test_binary,
    }

    return {
        "lstm":        lstm,
        "gru":         gru,
        "transformer": tst,
        "rf":          rf,
        "xgb":         xgb,
        "svm":         svm,
    }
