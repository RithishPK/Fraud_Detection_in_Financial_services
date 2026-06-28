# 💳 Credit Card Fraud Detection Dashboard

![](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/Data_Scaling-RobustScaler-007ACC?style=for-the-badge)

An interactive, high-performance financial analytics web application engineered to visualize data patterns and deploy real-time machine learning inference engines to flag fraudulent transactions instantly.

---

## 📌 Project Overview
Financial institutions lose billions annually to payment card frauds. This project bridges data engineering and machine learning by training models on deeply imbalanced financial records (containing anonymized Principal Components $V1$ to $V28$) and serving predictions via a user-facing dashboard.

### Key Operational Challenges Addressed:
* **Massive Class Imbalance:** Employs advanced under-sampling mapping to cleanly balance legitimate and malicious occurrences.
* **Feature Skewness:** Implements outlier-resistant data transformation pipelines using robust calculations.
* **Real-time Deployment:** Hosts a simulated sandbox letting developers input custom transactional telemetry matrices to test production models instantly.

---

## 🛠️ The Analytical Workflow
The application handles the complete end-to-end data lifecycle:



1. **Ingestion:** Raw transaction tracking sequences are uploaded securely via the web portal.
2. **Anti-Leakage Partitioning:** Splits raw data frames into isolated validation partitions before any modifications are executed.
3. **Robust Scaling:** Uses IQR scaling metrics mapping to process wildly fluctuating data inputs cleanly.
4. **Ensemble Classification:** Evaluates profiles against weighted Logistic Regression or Random Forest structures.

---

## 💻 Technical Architecture & Stack

### **Data & UI Engineering**
* **Streamlit (`UI Layer`):** Controls interactive widget updates and provides active persistent memory via stateful thread tracking (`st.session_state`).
* **Pandas & NumPy (`Core Processing`):** Manages quick indexing, vector groupings, and operational dataframe generation.

### **Machine Learning Suite**
* **Scikit-Learn (`Modeling`):** Manages training arrays, split partitions, standard scoring calculations, and classification matrix reports.
* **Matplotlib & Seaborn (`Data Viz`):** Generates distribution histograms, data correlations, and predictive curve mappings.

---

## 🚀 Interactive Feature Tabs

| Application Interface Tab | Operational Capability Provided |
| :--- | :--- |
| **📘 Data Overview** | Displays null evaluations, shape properties, summary trends, and interactive dataset distributions. |
| **📊 Visualization Dashboard** | Provides automated correlation heatmaps alongside scaled density curves. |
| **🤖 Model Training** | Trains selectable models on the fly, saving weights and active scalers directly to memory cache structures. |
| **🔮 Live Inference** | Allows dynamic variable tweaking (Time, Amount, and all 28 PCA dimensions) to extract instant fraud vectors. |

---

## 📦 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RithishPK/Fraud_Detection_in_Financial_services.git](https://github.com/RithishPK/Fraud_Detection_in_Financial_services.git)
   cd Fraud_Detection_in_Financial_services/Streamlit

Install system dependencies:

Bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
Incorporate Data Assets:

Download your source dataset (e.g., the standard Kaggle creditcard.csv log bundle).

Place the creditcard.csv file directly inside your local Streamlit/ subfolder.
(Note: The embedded .gitignore setting keeps this massive file safely hidden from your public git tracking histories).

Launch the Engine:

Bash
streamlit run app.py
📈 Sample Inference Preview
When testing raw transactional parameters inside the sandbox, the model outputs dynamic indicators:

🔴 High Risk Flagged: Triggered instantly if calculation confidence vectors break past 50% threat capacity thresholds.

🟢 Approved State: Verified if structural parameter variances stay within standard historical metrics.

Developed by RithishPK • Connect for reviews or expansion inquiries.
