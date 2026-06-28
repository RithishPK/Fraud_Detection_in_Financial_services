import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# Initialize session state for the model, scaler, and features
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'feature_columns' not in st.session_state:
    st.session_state['feature_columns'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None

# -------------------------------
# BACKGROUND IMAGE (DIMMED)
# -------------------------------
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            inset: 0;
            background-color: rgba(0,0,0,0.7);
        }}
        h1, h2, h3, h4, h5, h6, p, div, label {{
            color: white !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

add_bg_from_local("credit-card.jpg")

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("#### Explore, Visualize, and Train Machine Learning Models for Fraud Detection")
st.write("---")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (creditcard.csv)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📘 DATA OVERVIEW", 
        "📊 VISUALIZATION DASHBOARD", 
        "🤖 MODEL TRAINING", 
        "🔮 LIVE INFERENCE"
    ])

    # ====================================================
    # TAB 1: DATA OVERVIEW
    # ====================================================
    with tab1:
        st.subheader("📊 Dataset Information")
        st.write(f"**Dataset Shape:** {data.shape}")
        st.write("Missing values in dataset:", data.isnull().sum().sum())

        st.write("### Class Distribution (Interactive)")
        class_counts = data['Class'].value_counts().rename({0: 'Legit (0)', 1: 'Fraud (1)'})
        st.bar_chart(class_counts)

        st.write("### 🔍 Preview of Dataset")
        st.dataframe(data.head())

        st.write("### ⚙️ Summary Statistics (Amount column)")
        st.write(data['Amount'].describe())

        st.markdown("✅ **Balanced Data Info (After Under-Sampling)**")
        st.write(balanced_data['Class'].value_counts())

    # ====================================================
    # TAB 2: VISUALIZATION DASHBOARD
    # ====================================================
    with tab2:
        st.subheader("📊 Data Visualization Dashboard")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            sns.histplot(legit['Amount'], bins=30, label='Legit', color='green', kde=True, ax=ax_hist)
            sns.histplot(fraud['Amount'], bins=30, label='Fraud', color='red', kde=True, ax=ax_hist)
            ax_hist.set_title("Transaction Amount Distribution")
            ax_hist.legend()
            st.pyplot(fig_hist, use_container_width=True)

        with col2:
            fig_bal, ax_bal = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Class', data=balanced_data, palette='viridis', ax=ax_bal)
            ax_bal.set_title("Balanced Dataset Class Count")
            st.pyplot(fig_bal, use_container_width=True)

    # ====================================================
    # TAB 3: MODEL TRAINING
    # ====================================================
    with tab3:
        st.subheader("🤖 Train and Evaluate Models")

        # Fixed syntax to prevent the duplicate axis/columns ValueError
        X = data.drop(columns='Class') 
        Y = data['Class']
        
        X_train_raw, X_test, Y_train_raw, Y_test = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=42
        )
        
        # Balance training data to prevent leakage validation issues
        train_legit = X_train_raw[Y_train_raw == 0]
        train_fraud = X_train_raw[Y_train_raw == 1]
        train_legit_sample = train_legit.sample(n=len(train_fraud), random_state=42)
        
        X_train = pd.concat([train_legit_sample, train_fraud], axis=0)
        Y_train = pd.concat([pd.Series(0, index=train_legit_sample.index), pd.Series(1, index=train_fraud.index)], axis=0)

        # Scale features using RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_choice = st.selectbox("Select Model", ("Logistic Regression", "Random Forest"))

        if st.button("🚀 Train Model"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)

            model.fit(X_train_scaled, Y_train)
            
            # Persist variables inside the UI session lifecycle
            st.session_state['trained_model'] = model
            st.session_state['feature_columns'] = list(X.columns)
            st.session_state['scaler'] = scaler

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_acc = accuracy_score(Y_train, y_train_pred)
            test_acc = accuracy_score(Y_test, y_test_pred)

            st.success(f"✅ Model Trained! Testing Accuracy: {test_acc:.4f}")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("📉 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                sns.heatmap(confusion_matrix(Y_test, y_test_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                st.pyplot(fig_cm)
            with col_m2:
                st.subheader("📋 Classification Report")
                st.text(classification_report(Y_test, y_test_pred))

    # ====================================================
    # TAB 4: LIVE INFERENCE
    # ====================================================
    with tab4:
        st.subheader("🔮 Analyze a New Transaction")
        
        if st.session_state['trained_model'] is None:
            st.warning("⚠️ Please train a machine learning model on **TAB 3** before running individual predictions.")
        else:
            st.markdown("Enter transaction parameters below to test your operational fraud classifications:")
            
            col_inp1, col_inp2, col_inp3 = st.columns(3)
            
            with col_inp1:
                input_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=25.0, step=1.0)
                input_time = st.number_input("Time (Seconds elapsed from first tx)", min_value=0.0, value=0.0, step=1.0)
            
            v_inputs = {}
            with col_inp2:
                st.markdown("**Principal Components (V1 - V14)**")
                for i in range(1, 15):
                    v_inputs[f'V{i}'] = st.slider(f'V{i}', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
                    
            with col_inp3:
                st.markdown("**Principal Components (V15 - V28)**")
                for i in range(15, 29):
                    v_inputs[f'V{i}'] = st.slider(f'V{i}', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

            input_data_dict = {'Time': input_time, 'Amount': input_amount}
            input_data_dict.update(v_inputs)
            
            input_df = pd.DataFrame([input_data_dict])[st.session_state['feature_columns']]
            
            st.write("---")
            if st.button("🔍 Analyze Transaction Validity"):
                input_scaled = st.session_state['scaler'].transform(input_df)
                prediction = st.session_state['trained_model'].predict(input_scaled)[0]
                
                if hasattr(st.session_state['trained_model'], "predict_proba"):
                    probabilities = st.session_state['trained_model'].predict_proba(input_scaled)[0]
                    fraud_probability = probabilities[1] * 100
                else:
                    fraud_probability = None

                if prediction == 1:
                    st.error(f"🚨 **Alert: High Risk of Fraud Detected!**")
                    if fraud_probability is not None:
                        st.metric(label="Fraud Probability Level", value=f"{fraud_probability:.2f}%", delta="⚠️ Critical Risk")
                else:
                    st.success(f"🟢 **Transaction Approved: Legitimate Activity Verified.**")
                    if fraud_probability is not None:
                        st.metric(label="Fraud Probability Level", value=f"{fraud_probability:.2f}%", delta="- Safe Vector")

else:
    st.warning("👆 Upload `creditcard.csv` to begin analysis.")