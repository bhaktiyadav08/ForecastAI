# 🏬 Retail Demand Intelligence & Sales Forecasting System

An end-to-end predictive analytics framework and interactive operations hub built to transition retail supply chains from reactive inventory planning to proactive demand management. This system processes multi-year transactional data to isolate seasonal buying cycles, track operational anomalies, segment product catalogs, and generate multi-period forward forecasts.

🔗 **Live Interative Dashboard:** `https://forecast-ai.streamlit.app/`  

---

## 🎯 Business Architecture & System ROI
This framework provides actionable strategic indicators for logistics executives and financial officers:
*   **Fulfillment Speed Strategy:** Identifies a steady **3.96-day fulfillment latency** baseline across all operational distribution lanes, signaling that geographic regional location does not impact dispatch velocity.
*   **Capital Optimization:** Isolates **Low Volume, High Volatility** inventory sub-categories, providing a data-backed directive to transition these specific assets to a lean *Just-In-Time (JIT)* vendor routing model to free up frozen cash reserves.
*   **Safety Stock Calibration:** Automates real-time demand acceleration alerts, signaling when high-velocity product groups require immediate safety cap expansion to eliminate warehouse stockout risks.

---

## 🛠️ Technology Stack & Engine Infrastructure
*   **Core Systems:** Python 3.11+, Pandas, NumPy, Scikit-learn
*   **Statistical Time-Series Engine:** SARIMAX `(1,1,1)x(1,1,1)12` modeling seasonality trend envelopes
*   **Machine Learning Classifiers:** Isolation Forest and Interquartile Range (IQR) rolling percentile boundaries for anomaly detection matrices
*   **Data UI Architecture:** Streamlit Web Application Framework wrapped with Plotly Express interactive graphics engine canvas

---

## 📂 Repository File Blueprint
```text
├── 📂 charts/                        # Automated static PNG graphic exports
│   ├── anomaly_detection_map.png    # Statistical IQR vs Isolation Forest scatter tracking
│   ├── monthly_sales_trend.png      # 4-Year macro transactional volume timeline
│   └── sarima_forecast.png          # Out-of-sample forward demand projection track
├── 📄 analysis.ipynb                 # Deep Data Engineering, Stationarity Testing & ML Modeling
├── 📄 app.py                         # Production-grade Plotly-powered Streamlit Dashboard
├── 📄 requirements.txt               # System dependencies configuration manifest
├── 📄 summary.pdf                    # 2-Page Executive C-Suite Brief (CFO & Supply Chain Head)
└── 📄 train.csv                      # Core transactional Superstore dataset
```

---

## ⚙️ Local Installation & Deployment

1. **Clone the Enterprise Workspace Repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. **Verify System Requirements Setup:**
   Ensure your Python virtual environment or active interpreter environment is initialized, then build your system package baseline framework:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Operational Control Tower App Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 🔮 Forecasting Engine Performance Accuracy
The core forecasting pipeline evaluates multi-period models, selecting a seasonal **SARIMAX configuration** to track structural Q4 consumer velocity spikes securely. 
*   **Evaluation Baseline Matrix:** System functions at an verified **87.6% Evaluation Accuracy Score Baseline**.
*   **Planning Horizon Margin Guardrails:** Applies tight predictive margin envelopes (**±12%** for Month 1; scaling out to **±18%** for Month 3) to safely handle volatile operational variance patterns.
