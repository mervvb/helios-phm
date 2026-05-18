# HELIOS PHM — Predictive Health Monitoring System
### NASA CMAPSS FD001 | RUL Tahmini + Arıza Tespiti

---

## Proje Yapısı

```
helios_phm/
├── backend/
│   ├── app.py              ← Flask API (tüm modeller burada)
│   └── requirements.txt    ← Python bağımlılıkları
├── frontend/
│   └── index.html          ← Tek dosya frontend (API'ye bağlanır)
├── data/                   ← CMAPSS veri dosyaları buraya
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── README.md
```

---

## Kurulum

### 1. Python ortamı
```bash
cd backend
pip install -r requirements.txt
```

### 2. Veri dosyaları (opsiyonel)
NASA CMAPSS FD001 veri setini indirin:
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` dosyalarını
`data/` klasörüne koyun.

> **Not:** Veri yoksa sistem otomatik olarak sentetik demo verisi üretir.

### 3. Backend'i başlat
```bash
cd backend
python app.py
```
Backend `http://localhost:5050` adresinde çalışır.

### 4. Frontend'i aç
`frontend/index.html` dosyasını tarayıcıda açın.
Veya backend üzerinden:
```
http://localhost:5050
```

---

## API Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET  | `/api/status` | Backend durumu, eğitilmiş modeller |
| POST | `/api/load_data` | Veriyi yükle (body: `{data_dir}`) |
| POST | `/api/train` | Model eğit (body: `{model: "all"\|"svm"\|"rf"\|"gru"\|"lstm"\|"xgb"}`) |
| GET  | `/api/results` | Tüm model sonuçları |
| GET  | `/api/units` | Tüm birimlerin risk durumu |
| GET  | `/api/unit/<id>` | Belirli birim detayı + sensör verileri |
| GET  | `/api/comparison` | Model karşılaştırma tablosu |
| GET  | `/api/predictions/<model>` | Model tahminleri vs gerçek değerler |
| POST | `/api/predict_single` | Anlık tek birim tahmini |
| GET  | `/api/sensor_stats` | Sensör istatistikleri |

---

## Modeller

| Model | Tip | Kaynak Notebook |
|-------|-----|-----------------|
| **SVM (RBF)** | RUL Regresyon | `svm.pdf` |
| **Random Forest** | Arıza Sınıflandırma | `demo-rf.ipynb` |
| **GRU** | RUL Regresyon | `gru_modelleme.pdf` |
| **LSTM** | RUL Regresyon | `LSTM_RUL_Tahmini_ipynb.pdf` |
| **XGBoost** | RUL + Arıza | `XGBoost_arıza.pdf` |

> **Not:** TensorFlow/tsai gerektiren LSTM ve GRU modelleri
> sklearn ile simüle edilmiştir. Gerçek derin öğrenme için
> aşağıdaki ek bağımlılıkları kurun:
> ```
> pip install tensorflow tsai torch
> ```

---

## Özellikler

- **Canlı Dashboard** — Risk seviyesi, RUL tahmini, arıza olasılığı
- **3D Motor Animasyonu** — T-700/CT7 türbin modeli
- **Model Karşılaştırma** — RMSE, MAE, R², F1, AUC, Confusion Matrix
- **Bakım Öncelik Kuyruğu** — Risk sırasına göre sıralanmış birimler
- **Sensör İzleme** — 14 aktif sensör kanalı
- **Otomatik Risk Sınıflandırma** — Yüksek / Orta / Düşük
- **Gerçek Zamanlı API** — 30s'de bir otomatik yenileme

---

## Kullanım

1. Backend'i başlatın: `python backend/app.py`
2. Tarayıcıda `frontend/index.html` açın
3. Sol panelde **▶ Train All Models** butonuna tıklayın
4. Eğitim tamamlandığında tüm grafikler ve tablolar dolar
5. Birimlere tıklayarak detay + sensör verilerini görün

---

## Gereksinimler

- Python 3.9+
- Flask, Flask-CORS
- NumPy, Pandas, Scikit-learn
- XGBoost (opsiyonel ama önerilir)
