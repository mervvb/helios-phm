"""
HELIOS PHM — Flask Backend  (v2 — demo modu dahil)
Demo modu: Gerçek veri yoksa NASA C-MAPSS FD001 gerçeğine yakın sonuçlar yüklenir.
"""

import sys, os, threading, warnings, csv as csv_lib
from io import StringIO
from datetime import datetime, timedelta
import numpy as np
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "rul_models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "fault_models"))

import lstm_model        as LSTM_MOD
import gru_model         as GRU_MOD
import transformer_model as TST_MOD
import random_forest     as RF_MOD
import xgboost_model     as XGB_MOD
import svm_model         as SVM_MOD

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder=os.path.join(ROOT_DIR, "frontend"), static_url_path="")
CORS(app)

RESULTS  = {}
TRAINING = {}
LOGS     = {}

ALLOWED_EXT = {".txt", ".csv", ".xlsx", ".xls", ".json"}
SAFETY_MARGIN = 5

MODEL_MAP = {
    "lstm":        ("RUL",   LSTM_MOD.train),
    "gru":         ("RUL",   GRU_MOD.train),
    "transformer": ("RUL",   TST_MOD.train),
    "rf":          ("Fault", RF_MOD.train),
    "xgb":         ("Fault", XGB_MOD.train),
    "svm":         ("Fault", SVM_MOD.train),
}

SENSOR_DESC = {
    "s2":  ("LPC Çıkış Sıcaklığı",       "thermal"),
    "s3":  ("HPC Çıkış Sıcaklığı",       "thermal"),
    "s4":  ("LPT Çıkış Sıcaklığı",       "thermal"),
    "s7":  ("HPC Çıkış Basıncı",         "pressure"),
    "s8":  ("Fiziksel Fan Hızı",          "vibration"),
    "s9":  ("Fiziksel Çekirdek Hızı",     "vibration"),
    "s11": ("HPC Statik Basınç",          "pressure"),
    "s12": ("Yakıt Akışı / Ps30",         "fuel"),
    "s13": ("Düzeltilmiş Fan Hızı",       "vibration"),
    "s14": ("Düzeltilmiş Çekirdek Hızı",  "vibration"),
    "s15": ("Bypass Oranı",               "thermal"),
    "s17": ("Bleed Entalpisi",            "pressure"),
    "s20": ("HP Türbin Soğutma Akışı",    "thermal"),
    "s21": ("LP Türbin Soğutma Akışı",    "thermal"),
}

ANOMALY_LABELS = {
    "thermal":   "Termal Anomali",
    "pressure":  "Basınç Anomalisi",
    "vibration": "Titreşim Anomalisi",
    "fuel":      "Yakıt Sistemi Anomalisi",
}

MAINTENANCE_ACTIONS = {
    "high":   ["Acil rulman muayenesi","Mil hizalama kontrolü","Soğutma sistemi testi","Kompresör kanat muayenesi","Türbin disk muayenesi"],
    "medium": ["48 saat içinde borescope muayenesi","Yağ analizi yapılmalı","Titreşim ölçümleri kontrol edilmeli","Soğutma kanalları temizlenmeli"],
    "low":    ["Rutin izleme devam etmeli","Bir sonraki zamanlanmış bakımda kontrol"],
}

MAINTENANCE_TYPE = {
    "high":   "Acil Bakım (Unscheduled)",
    "medium": "Planlı Erken Bakım",
    "low":    "Rutin Bakım",
}

PRODUCTION_IMPACT = {
    "high":   {"level": "High",   "downtime_h": 24, "cost_usd": 28000},
    "medium": {"level": "Medium", "downtime_h": 8,  "cost_usd": 8000},
    "low":    {"level": "Low",    "downtime_h": 2,  "cost_usd": 1500},
}

CRITICALITY = {"high": 1.0, "medium": 0.6, "low": 0.2}
TEAMS = ["Team A", "Team B"]


# ─────────────────────────────────────────────────────────────────────────
# DEMO MODU — ayrı dosya gerektirmez
# ─────────────────────────────────────────────────────────────────────────

def _load_demo_results():
    """NASA C-MAPSS FD001 gerçeğine yakın demo sonuçlar üretir."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix
    )
    rng = np.random.RandomState(42)

    actual_rul = np.array([
        112,98,69,82,91,93,91,95,111,96,
        97,124,95,107,83,84,95,61,87,94,
        102,111,113,98,125,119,105,91,100,101,
        100,63,104,78,101,19,82,125,91,117,
        81,75,30,20,66,125,77,125,61,93,
        83,77,86,11,93,87,20,98,90,6,
        11,80,91,78,93,18,30,93,10,5,
        100,1,95,94,26,101,88,91,26,49,
        28,1,20,58,3,52,25,36,29,37,
        45,28,9,23,45,40,46,45,6,9,
    ], dtype=float)
    actual_rul = np.clip(actual_rul, 0, 125).tolist()
    n = len(actual_rul)

    def noisy(vals, sigma=5.5):
        return [max(0.0, float(v) + float(rng.normal(0, sigma))) for v in vals]

    # LSTM
    lstm_preds = noisy(actual_rul, 6.2)
    lstm_val_a = [max(0, float(rng.normal(60, 28))) for _ in range(400)]
    lstm_val_p = noisy(lstm_val_a, 7.1)
    lstm_elog  = [{"epoch": e+1,
                   "loss":     round(0.08*(0.92**e)+0.012, 4),
                   "val_loss": round(0.09*(0.91**e)+0.015, 4),
                   "mae":      round(12.0*(0.93**e)+4.0,   4)}
                  for e in range(32)]
    lstm = {
        "name":"LSTM","type":"regression",
        "mae":14.23,"rmse":19.87,"r2":0.8612,
        "val_mae":14.89,"val_rmse":20.31,"val_r2":0.8534,
        "duration_s":48.3,"epochs":32,"epoch_log":lstm_elog,
        "predictions":lstm_preds,"val_predictions":lstm_val_p,
        "val_actual":lstm_val_a,"actual":actual_rul,"results_df":[],
    }

    # GRU
    gru_preds = noisy(actual_rul, 5.4)
    gru_val_a = [max(0, float(rng.normal(60, 27))) for _ in range(400)]
    gru_val_p = noisy(gru_val_a, 6.3)
    gru_elog  = [{"epoch": e+1,
                  "loss":     round(0.075*(0.91**e)+0.011, 4),
                  "val_loss": round(0.085*(0.90**e)+0.014, 4)}
                 for e in range(38)]
    gru = {
        "name":"GRU","type":"regression",
        "mae":13.41,"rmse":18.54,"r2":0.8743,
        "val_mae":14.02,"val_rmse":19.17,"val_r2":0.8651,
        "duration_s":41.7,"epochs":38,"epoch_log":gru_elog,
        "predictions":gru_preds,"val_predictions":gru_val_p,
        "val_actual":gru_val_a,"actual":actual_rul,
    }

    # TST
    tst_preds = noisy(actual_rul, 7.3)
    tst = {
        "name":"TST (tsai)","type":"regression",
        "mae":15.67,"rmse":21.93,"r2":0.8401,
        "duration_s":118.3,"epochs":60,
        "train_curve":[round(0.082*(0.93**e)+0.013,4) for e in range(60)],
        "valid_curve":[round(0.091*(0.92**e)+0.016,4) for e in range(60)],
        "predictions":tst_preds,"actual":actual_rul,
        "params":{"window":50,"d_model":128,"n_heads":8,"d_ff":512,
                  "dropout":0.2,"n_layers":3,"epochs":60,"lr":0.001},
    }

    # RF
    y_bin = [1 if v <= 30 else 0 for v in actual_rul]
    rf_probs = [min(1.0, max(0.0, (1-v/125)*0.83 + float(rng.normal(0,0.04))))
                for v in actual_rul]
    rf_preds = [1 if p >= 0.45 else 0 for p in rf_probs]
    rf = {
        "name":"Random Forest","type":"classification",
        "f1": 0.8214, "precision": 0.7931, "recall": 0.8519, "auc": 0.9312,
        "threshold":0.45,
        "confusion_matrix":confusion_matrix(y_bin,rf_preds).tolist(),
        "feature_importance":{"s12_trend":0.089,"s11_son":0.071,"s4_ort":0.063,
                               "s14_std":0.058,"s9_trend":0.052,"s7_max":0.048,
                               "s2_ort":0.044,"s3_son":0.041,"s20_trend":0.038,"s21_std":0.034},
        "duration_s":18.4,
        "accuracy":0.8900,
        "predictions":rf_preds,"probabilities":rf_probs,"actual":y_bin,
    }

    # XGBoost
    xgb_probs = [min(1.0, max(0.0, (1-v/125)*0.88 + float(rng.normal(0,0.05))))
                 for v in actual_rul]
    xgb_rul_p = noisy(actual_rul, 5.0)
    xgb_cls   = [1 if p >= 0.5 else 0 for p in xgb_probs]

    def _risk(prob, rul):
        if prob >= 0.70 or rul <= 30: return "Yüksek Risk"
        if prob >= 0.40 or rul <= 60: return "Orta Risk"
        return "Düşük Risk"
    def _mrec(risk):
        if risk == "Yüksek Risk": return "Acil bakım planına alınmalı"
        if risk == "Orta Risk":   return "Yakından izlenmeli ve bakım planına dahil edilmeli"
        return "Normal izleme devam etmeli"

    rdf = []
    for i,(r,p) in enumerate(zip(xgb_rul_p, xgb_probs)):
        r = max(0, r)
        rk = _risk(p, r)
        rdf.append({"unit_number":i+1,"time_cycle":200+i,
                    "predicted_RUL":round(r,1),
                    "failure_probability_percent":round(p*100,2),
                    "risk_level":rk,"maintenance_recommendation":_mrec(rk)})
    xgb = {
        "name":"XGBoost","type":"both",
        "accuracy":0.8800,
        "precision": 0.8571, "recall": 0.8704, "f1": 0.8637,
        "auc": 0.9487, "auc_test": 0.9401,
        "confusion_matrix":confusion_matrix(y_bin,xgb_cls).tolist(),
        "duration_s":22.8,
        "predictions":xgb_rul_p,"probabilities":xgb_probs,
        "actual":actual_rul,"results_df":rdf,
    }

    # SVM
    svm_probs = [min(1.0, max(0.0, (1-v/125)*0.82 + float(rng.normal(0,0.04))))
                 for v in actual_rul]
    svm_preds = [1 if p >= 0.5 else 0 for p in svm_probs]
    svm_f1, svm_prec, svm_rec, svm_auc = 0.7843, 0.7612, 0.8092, 0.9074
    svm = {
        "name":"SVM (RBF)","type":"classification",
        "f1": svm_f1, "precision": svm_prec, "recall": svm_rec, "auc": svm_auc,
        "confusion_matrix":confusion_matrix(y_bin,svm_preds).tolist(),
        "best_params":{"C":1,"kernel":"rbf","class_weight":"balanced"},
        "duration_s":35.2,
        "predictions":svm_preds,"probabilities":svm_probs,"actual":y_bin,
    }

    return {"lstm":lstm,"gru":gru,"transformer":tst,"rf":rf,"xgb":xgb,"svm":svm}


def _init_demo():
    """Uygulama başlangıcında demo verileri yükle."""
    try:
        demo = _load_demo_results()
        RESULTS.update(demo)
        print("  ✓  Demo modu aktif — 6 model sonucu yüklendi")
    except Exception as e:
        print(f"  ⚠  Demo yüklenemedi: {e}")

# Her zaman demo yükle; gerçek eğitim üzerine yazar
_init_demo()


# ─────────────────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────────────────

def make_log_cb(key):
    def cb(msg):
        LOGS.setdefault(key, []).append(str(msg))
    return cb


def _run_model(key, train_fn):
    TRAINING[key] = True
    LOGS[key] = []
    try:
        result = train_fn(DATA_DIR, log_callback=make_log_cb(key))
        for k in ("model_obj","scaler_obj","learner"):
            result.pop(k, None)
        RESULTS[key] = result
    except Exception as e:
        RESULTS[key] = {"name": key, "error": str(e)}
    finally:
        TRAINING.pop(key, None)


def _analyze_failure_reasons(uid, df_test):
    grp = df_test[df_test["unit"] == uid].sort_values("cycle")
    if grp.empty or len(grp) < 5:
        return {"primary_anomalies":[],"degradation_rate":"unknown","sensor_alerts":[]}
    ANALYSIS_SENSORS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14"]
    last_n = grp.tail(10); first_n = grp.head(10)
    sensor_alerts = []; anomaly_cats = {}
    for s in ANALYSIS_SENSORS:
        if s not in grp.columns: continue
        desc, cat = SENSOR_DESC.get(s,(s,"unknown"))
        last_val=float(last_n[s].mean()); first_val=float(first_n[s].mean())
        std_val=float(grp[s].std())
        if std_val < 1e-6: continue
        mean_all=float(grp[s].mean())
        z_score=abs(last_val-mean_all)/(std_val+1e-8)
        trend_pct=((last_val-first_val)/(abs(first_val)+1e-8))*100
        if z_score > 2.0 or abs(trend_pct) > 5:
            direction="↑" if trend_pct>0 else "↓"
            sensor_alerts.append({"sensor":s,"name":desc,"category":cat,
                "z_score":round(z_score,2),"trend_pct":round(trend_pct,2),
                "direction":direction,"severity":"critical" if z_score>3.0 else "warning"})
            anomaly_cats[cat]=anomaly_cats.get(cat,0)+z_score
    sorted_cats=sorted(anomaly_cats.items(),key=lambda x:x[1],reverse=True)
    primary=[{"category":ANOMALY_LABELS.get(c,c),"score":round(s,2)} for c,s in sorted_cats[:3]]
    deg_rate="unknown"
    if "s12" in grp.columns and len(grp)>=8:
        last5=float(grp.tail(5)["s12"].mean())
        prev5=float(grp.iloc[-10:-5]["s12"].mean()) if len(grp)>=10 else float(grp.head(5)["s12"].mean())
        delta=abs(last5-prev5)
        deg_rate="stable" if delta<0.5 else ("moderate" if delta<2.0 else "rapid")
    sensor_alerts.sort(key=lambda x:x["z_score"],reverse=True)
    return {"primary_anomalies":primary,"degradation_rate":deg_rate,"sensor_alerts":sensor_alerts[:5]}


def _priority_score(risk, rul, prob):
    risk_score = {"high":1.0,"medium":0.6,"low":0.2}.get(risk,0.2)
    crit_score = CRITICALITY.get(risk,0.2)
    prod_score = {"high":1.0,"medium":0.55,"low":0.15}.get(risk,0.15)
    prob_factor=min(1.0,prob/100); rul_factor=max(0.0,1.0-(rul/125))
    weighted_risk=risk_score*0.4+prob_factor*0.3+rul_factor*0.3
    score=round(0.5*weighted_risk+0.3*crit_score+0.2*prod_score,4)
    return {"score":round(score,3),"score_pct":round(score*100,1),
            "risk_component":round(weighted_risk,3),"crit_component":round(crit_score,3),
            "prod_component":round(prod_score,3),
            "formula":"Priority = 0.5 × Risk + 0.3 × Criticality + 0.2 × ProductionImpact"}


def _load_test_df():
    try:
        import pandas as pd
        COL_NAMES=["unit","cycle","op1","op2","op3",
                   "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
                   "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"]
        path=os.path.join(DATA_DIR,"test_FD001.txt")
        if os.path.exists(path):
            return pd.read_csv(path,sep=r"\s+",header=None,names=COL_NAMES)
    except Exception: pass
    return None


def _get_units():
    source=RESULTS.get("xgb") or RESULTS.get("rf")
    rul_src=RESULTS.get("lstm") or RESULTS.get("gru") or {}
    rul_preds=rul_src.get("predictions",[])
    units=[]
    if source and "results_df" in source:
        for row in source["results_df"]:
            rul=float(row.get("predicted_RUL",60))
            prob=float(row.get("failure_probability_percent",0))
            risk_raw=row.get("risk_level","")
            risk="high" if "Yüksek" in risk_raw else ("medium" if "Orta" in risk_raw else "low")
            ps=_priority_score(risk,rul,prob)
            units.append({"unit":int(row.get("unit_number",0)),"rul":round(rul,1),
                          "prob":round(prob,2),"risk":risk,"priority_score":ps["score_pct"]})
    elif source:
        probs=source.get("probabilities",[])
        for i,prob in enumerate(probs):
            rul=float(rul_preds[i]) if i<len(rul_preds) else 60.0
            prob_pct=float(prob)*100
            risk=("high" if (prob_pct>=70 or rul<=30) else
                  "medium" if (prob_pct>=40 or rul<=60) else "low")
            ps=_priority_score(risk,rul,prob_pct)
            units.append({"unit":i+1,"rul":round(rul,1),"prob":round(prob_pct,2),
                          "risk":risk,"priority_score":ps["score_pct"]})
    return sorted(units,key=lambda x:x["priority_score"],reverse=True)


def _get_plan(units):
    plan=[]
    for idx,u in enumerate(units):
        maint_window_cycles=max(0,u["rul"]-SAFETY_MARGIN)
        prod=PRODUCTION_IMPACT.get(u["risk"],PRODUCTION_IMPACT["low"])
        team=TEAMS[idx%len(TEAMS)]
        plan.append({**u,"safety_margin":SAFETY_MARGIN,
                     "maintenance_window_cycles":round(maint_window_cycles,1),
                     "maintenance_type":MAINTENANCE_TYPE.get(u["risk"],"Rutin Bakım"),
                     "actions":MAINTENANCE_ACTIONS.get(u["risk"],[]),
                     "team":team,"production_impact":prod,
                     "priority_score":u.get("priority_score",0)})
    return plan


# ─────────────────────────────────────────────────────────────────────────
# ENDPOINT'LER
# ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    return send_from_directory(os.path.join(ROOT_DIR, "frontend"), "index.html")


@app.route("/api/status")
def api_status():
    data_ready=os.path.exists(os.path.join(DATA_DIR,"train_FD001.txt"))
    return jsonify({"data_ready":data_ready,
                    "demo_mode": not data_ready and bool(RESULTS),
                    "models_trained":list(RESULTS.keys()),
                    "models_training":list(TRAINING.keys()),
                    "available_models":list(MODEL_MAP.keys())})


@app.route("/api/train", methods=["POST"])
def api_train():
    model_name=request.json.get("model","all")
    targets=list(MODEL_MAP.keys()) if model_name=="all" else [model_name]
    started=[]
    for key in targets:
        if key not in MODEL_MAP or key in TRAINING: continue
        _,train_fn=MODEL_MAP[key]
        threading.Thread(target=_run_model,args=(key,train_fn),daemon=True).start()
        started.append(key)
    return jsonify({"status":"training_started","started":started,
                    "message":f"{', '.join(started)} eğitimi başladı."})


@app.route("/api/train_sync", methods=["POST"])
def api_train_sync():
    key=request.json.get("model","rf")
    if key not in MODEL_MAP: return jsonify({"error":"Bilinmeyen model"}),400
    if key in TRAINING: return jsonify({"error":"Zaten eğitiliyor"}),409
    _,train_fn=MODEL_MAP[key]
    _run_model(key,train_fn)
    return jsonify({"status":"ok","result":RESULTS.get(key,{})})


@app.route("/api/logs/<model_key>")
def api_logs(model_key):
    return jsonify({"model":model_key,"logs":LOGS.get(model_key,[]),
                    "training":model_key in TRAINING,"done":model_key in RESULTS})


@app.route("/api/results")
def api_results():
    return jsonify(RESULTS)


@app.route("/api/results/<model_key>")
def api_result_single(model_key):
    if model_key not in RESULTS:
        return jsonify({"error":"Model henüz eğitilmedi"}),404
    return jsonify(RESULTS[model_key])


@app.route("/api/comparison")
def api_comparison():
    rows=[]
    for key,r in RESULTS.items():
        row={"model":r.get("name",key),"key":key}
        for m in ("rmse","mae","r2","f1","auc","accuracy","precision","recall","duration_s"):
            if m in r: row[m]=r[m]
        rows.append(row)
    return jsonify(rows)


@app.route("/api/units")
def api_units():
    units=_get_units()
    high=sum(1 for u in units if u["risk"]=="high")
    medium=sum(1 for u in units if u["risk"]=="medium")
    low=sum(1 for u in units if u["risk"]=="low")
    return jsonify({"units":units,"summary":{"high":high,"medium":medium,"low":low,"total":len(units)}})


@app.route("/api/predictions/<model_key>")
def api_predictions(model_key):
    if model_key not in RESULTS:
        return jsonify({"error":"Model henüz eğitilmedi"}),404
    r=RESULTS[model_key]
    return jsonify({"model":r.get("name",model_key),"predictions":r.get("predictions",[]),
                    "actual":r.get("actual",[]),"val_predictions":r.get("val_predictions",[]),
                    "val_actual":r.get("val_actual",[]),"probabilities":r.get("probabilities",[])})


@app.route("/api/unit/<int:uid>")
def api_unit_detail(uid):
    import pandas as pd
    source=RESULTS.get("xgb") or RESULTS.get("rf")
    rul_src=RESULTS.get("lstm") or RESULTS.get("gru") or {}
    rul_preds=rul_src.get("predictions",[])
    unit_data=None
    if source and "results_df" in source:
        for row in source["results_df"]:
            if int(row.get("unit_number",0))==uid:
                rul=float(row.get("predicted_RUL",50))
                prob=float(row.get("failure_probability_percent",0))
                risk_raw=row.get("risk_level","")
                risk="high" if "Yüksek" in risk_raw else ("medium" if "Orta" in risk_raw else "low")
                unit_data={"unit":uid,"rul":round(rul,1),"prob":round(prob,2),"risk":risk}
                break
    elif source:
        probs=source.get("probabilities",[])
        for i,prob in enumerate(probs):
            if i+1==uid:
                rul=float(rul_preds[i]) if i<len(rul_preds) else 60.0
                prob_pct=float(prob)*100
                risk=("high" if (prob_pct>=70 or rul<=30) else
                      "medium" if (prob_pct>=40 or rul<=60) else "low")
                unit_data={"unit":uid,"rul":round(rul,1),"prob":round(prob_pct,2),"risk":risk}
                break
    if not unit_data:
        idx=uid-1
        if rul_preds and 0<=idx<len(rul_preds):
            rul=float(rul_preds[idx])
            unit_data={"unit":uid,"rul":round(rul,1),"prob":0.0,"risk":"low"}
        else:
            return jsonify({"error":"Birim bulunamadı"}),404
    sensors={}
    df_test=_load_test_df()
    if df_test is not None:
        grp=df_test[df_test["unit"]==uid]
        if not grp.empty:
            last=grp.sort_values("cycle").iloc[-1]
            for s in ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14"]:
                if s in last.index: sensors[s]=round(float(last[s]),4)
    failure_analysis={}
    if df_test is not None:
        failure_analysis=_analyze_failure_reasons(uid,df_test)
    ps=_priority_score(unit_data["risk"],unit_data["rul"],unit_data["prob"])
    maint_window_cycles=max(0,unit_data["rul"]-SAFETY_MARGIN)
    actions=MAINTENANCE_ACTIONS.get(unit_data["risk"],[])
    maint_type=MAINTENANCE_TYPE.get(unit_data["risk"],"Rutin Bakım")
    prod_impact=PRODUCTION_IMPACT.get(unit_data["risk"],PRODUCTION_IMPACT["low"])
    return jsonify({**unit_data,"sensors":sensors,"failure_analysis":failure_analysis,
                    "priority":ps,"maintenance":{"window_cycles":round(maint_window_cycles,1),
                    "safety_margin":SAFETY_MARGIN,"type":maint_type,"actions":actions,
                    "note":"C-MAPSS cycle = aşınma adımı, fiziksel zaman değil"},
                    "production_impact":prod_impact})


@app.route("/api/maintenance_plan")
def api_maintenance_plan():
    units=_get_units(); plan=_get_plan(units)
    today=datetime.now().strftime("%Y-%m-%d")
    total_downtime=sum(p["production_impact"]["downtime_h"] for p in plan)
    total_cost_save=sum(p["production_impact"]["cost_usd"] for p in plan)
    team_schedule={}
    for p in plan: team_schedule.setdefault(p["team"],[]).append(f"#{p['unit']}")
    return jsonify({"today":today,"safety_margin":SAFETY_MARGIN,"plan":plan,
                    "fleet_summary":{"total_units":len(plan),
                    "critical":sum(1 for p in plan if p["risk"]=="high"),
                    "estimated_downtime_h":total_downtime,
                    "potential_cost_saved":total_cost_save,"team_schedule":team_schedule}})


@app.route("/api/fleet_heatmap")
def api_fleet_heatmap():
    units=_get_units(); rows=[]
    for u in units:
        ps=_priority_score(u["risk"],u["rul"],u["prob"])
        rows.append({"unit":u["unit"],"rul":u["rul"],"prob":u["prob"],"risk":u["risk"],
                     "priority_score":ps["score_pct"],
                     "color_bin":("critical" if u["rul"]<20 else "high" if u["rul"]<40
                                  else "medium" if u["rul"]<60 else "low")})
    return jsonify({"units":rows})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "files" not in request.files:
        return jsonify({"status":"error","message":"Dosya bulunamadı"}),400
    uploaded,errors=[],[]
    for f in request.files.getlist("files"):
        if not f.filename: continue
        ext=os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            errors.append(f"{f.filename}: desteklenmeyen format ({ext})"); continue
        fname=secure_filename(f.filename)
        os.makedirs(DATA_DIR,exist_ok=True)
        f.save(os.path.join(DATA_DIR,fname))
        uploaded.append(fname)
    if not uploaded:
        return jsonify({"status":"error","message":"Hiçbir dosya yüklenmedi. "+";"
                        .join(errors)}),400
    return jsonify({"status":"ok","uploaded":uploaded,"errors":errors,
                    "message":f"{len(uploaded)} dosya yüklendi: {', '.join(uploaded)}"})


@app.route("/api/export/excel")
def api_export_excel():
    units=_get_units(); plan=_get_plan(units)
    buf=StringIO(); w=csv_lib.writer(buf)
    w.writerow(["HELIOS PHM — Analiz Raporu"])
    w.writerow(["Tarih",datetime.now().strftime("%d/%m/%Y %H:%M")]); w.writerow([])
    w.writerow(["MOTOR DURUMU"])
    w.writerow(["Motor","Risk","RUL (cycle)","Arıza Olasılığı (%)","Priority Score"])
    for u in units:
        w.writerow([f"#{u['unit']}",u["risk"].upper(),u["rul"],u["prob"],u.get("priority_score","")])
    w.writerow([]); w.writerow(["MODEL PERFORMANSI"])
    w.writerow(["Model","Tip","R²","RMSE","MAE","F1","AUC","Precision","Recall"])
    for key,r in RESULTS.items():
        if key=="tst_exp": continue
        typ="RUL" if key in ("lstm","gru","transformer") else "Arıza"
        w.writerow([r.get("name",key.upper()),typ,r.get("r2",""),r.get("rmse",""),
                    r.get("mae",""),r.get("f1",""),r.get("auc",""),
                    r.get("precision",""),r.get("recall","")])
    fname=f"helios_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    content="\ufeff"+buf.getvalue()
    return Response(content,mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition":f'attachment; filename="{fname}"',
                             "Access-Control-Expose-Headers":"Content-Disposition"})


@app.route("/api/export/pdf")
def api_export_pdf():
    units=_get_units(); plan=_get_plan(units)
    def risk_col(r):
        return "#c62828" if r=="high" else ("#f57c00" if r=="medium" else "#00897b")
    unit_rows="".join(
        f'<tr><td>#{u["unit"]}</td>'
        f'<td style="color:{risk_col(u["risk"])};font-weight:700">{u["risk"].upper()}</td>'
        f'<td>{u["rul"]}</td><td>{u["prob"]}%</td>'
        f'<td style="font-weight:700">{u.get("priority_score",0)}</td></tr>'
        for u in units
    ) or "<tr><td colspan=5 style='text-align:center;color:#7a909f'>Veri yok</td></tr>"
    model_rows=""
    for key,r in RESULTS.items():
        if key=="tst_exp": continue
        typ="RUL" if key in ("lstm","gru","transformer") else "Arıza"
        model_rows+=(f'<tr><td>{r.get("name",key.upper())}</td><td>{typ}</td>'
                     f'<td>{r.get("r2","—")}</td><td>{r.get("rmse","—")}</td>'
                     f'<td>{r.get("mae","—")}</td><td>{r.get("f1","—")}</td>'
                     f'<td>{r.get("auc","—")}</td></tr>')
    if not model_rows:
        model_rows="<tr><td colspan=7 style='text-align:center;color:#7a909f'>Veri yok</td></tr>"
    plan_rows="".join(
        f'<tr><td>#{p["unit"]}</td>'
        f'<td style="color:{risk_col(p["risk"])};font-weight:700">{p["risk"].upper()}</td>'
        f'<td>{p["rul"]}</td>'
        f'<td>{p.get("maintenance_window_cycles","—")}</td>'
        f'<td>{p["team"]}</td>'
        f'<td>{p["production_impact"]["downtime_h"]}h</td>'
        f'<td>${p["production_impact"]["cost_usd"]:,}</td></tr>'
        for p in plan[:20]
    ) or "<tr><td colspan=7 style='text-align:center;color:#7a909f'>Veri yok</td></tr>"
    fs={"total":len(plan),"critical":sum(1 for p in plan if p["risk"]=="high"),
        "downtime":sum(p["production_impact"]["downtime_h"] for p in plan),
        "cost":sum(p["production_impact"]["cost_usd"] for p in plan)}
    html=f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8"><title>HELIOS PHM Raporu</title>
<style>
  body{{font-family:Inter,Arial,sans-serif;padding:32px;color:#0f1923;font-size:13px;max-width:960px;margin:0 auto}}
  h1{{font-size:22px;font-weight:700;color:#0097a7;margin-bottom:4px}}
  h2{{font-size:15px;font-weight:600;margin:28px 0 10px;border-bottom:2px solid #e4e8ee;padding-bottom:6px}}
  .meta{{font-size:11px;color:#7a909f;margin-bottom:28px}}
  .summary{{display:flex;gap:16px;margin-bottom:24px}}
  .sum-card{{background:#f6f8fa;border:1px solid #e4e8ee;border-radius:6px;padding:12px 16px;flex:1;text-align:center}}
  .sum-val{{font-size:22px;font-weight:700;color:#0097a7}}
  .sum-lbl{{font-size:10px;color:#7a909f;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;margin-bottom:8px}}
  th{{background:#f6f8fa;text-align:left;padding:8px 10px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#7a909f;border-bottom:2px solid #d0d7de}}
  td{{padding:7px 10px;border-bottom:1px solid #e4e8ee;font-size:12px}}
  tr:nth-child(even) td{{background:#f6f8fa}}
  .btn{{padding:9px 22px;background:#0097a7;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;margin-top:28px}}
  @media print{{.btn{{display:none}}}}
</style></head><body>
<h1>HELIOS PHM — Analiz Raporu</h1>
<div class="meta">Oluşturma: {datetime.now().strftime("%d/%m/%Y %H:%M")} · Safety Margin: {SAFETY_MARGIN} cycle · Demo Modu</div>
<div class="summary">
  <div class="sum-card"><div class="sum-val">{fs["total"]}</div><div class="sum-lbl">Toplam Motor</div></div>
  <div class="sum-card"><div class="sum-val" style="color:#c62828">{fs["critical"]}</div><div class="sum-lbl">Kritik</div></div>
  <div class="sum-card"><div class="sum-val">{fs["downtime"]}h</div><div class="sum-lbl">Tahmini Kesinti</div></div>
  <div class="sum-card"><div class="sum-val">${fs["cost"]:,}</div><div class="sum-lbl">Önlenen Maliyet</div></div>
</div>
<h2>Motor Öncelik Listesi</h2>
<table><thead><tr><th>Motor</th><th>Risk</th><th>RUL</th><th>Arıza Olasılığı</th><th>Priority Score</th></tr></thead>
<tbody>{unit_rows}</tbody></table>
<h2>Model Performansı</h2>
<table><thead><tr><th>Model</th><th>Tip</th><th>R²</th><th>RMSE</th><th>MAE</th><th>F1</th><th>AUC</th></tr></thead>
<tbody>{model_rows}</tbody></table>
<h2>Bakım Planı</h2>
<table><thead><tr><th>Motor</th><th>Risk</th><th>RUL</th><th>Bakım Penceresi</th><th>Ekip</th><th>Kesinti</th><th>Önlenen Maliyet</th></tr></thead>
<tbody>{plan_rows}</tbody></table>
<button class="btn" onclick="window.print()">Yazdır / PDF Kaydet</button>
</body></html>"""
    return Response(html,mimetype="text/html; charset=utf-8")


@app.route("/api/tst_experiments", methods=["POST"])
def api_tst_experiments():
    def run():
        TRAINING["tst_exp"]=True
        try:
            results=TST_MOD.train_all_experiments(DATA_DIR,log_callback=make_log_cb("tst_exp"))
            for r in results: r.pop("model_obj",None); r.pop("learner",None)
            RESULTS["tst_exp"]=results
        except Exception as e:
            RESULTS["tst_exp"]={"error":str(e)}
        finally:
            TRAINING.pop("tst_exp",None)
    threading.Thread(target=run,daemon=True).start()
    return jsonify({"status":"started","message":"8 deney başladı."})


@app.route("/api/tst_experiments_result")
def api_tst_experiments_result():
    return jsonify(RESULTS.get("tst_exp",[]))


if __name__ == "__main__":
    print("="*60)
    print("  HELIOS PHM v2 — Backend  →  http://localhost:5050")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Safety margin : {SAFETY_MARGIN} cycle")
    print("="*60)
    required=["train_FD001.txt","test_FD001.txt","RUL_FD001.txt"]
    missing=[f for f in required if not os.path.exists(os.path.join(DATA_DIR,f))]
    if missing: print(f"  ⚠  Eksik: {missing} — Demo modu aktif")
    else: print("  ✓  Veri dosyaları hazır")
    print()
    app.run(debug=False, port=5050, host="0.0.0.0", threaded=True)
