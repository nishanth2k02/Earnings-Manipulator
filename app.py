import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Earnings Manipulator Detector", layout="wide")

st.title("📊 Earnings Manipulation Detection App")
st.markdown("""
This app detects potential earnings manipulation using the **Beneish M-Score variables** and Machine Learning.
""")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'Earnings Manipulator.xlsx'", type=["xlsx"])

# --- HELPER FUNCTIONS ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_pred)
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0
    }

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # --- 2. PREPROCESSING ---
        # Fixed features based on your project
        feature_cols = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
        target_col = 'Manipulator'

        # Check if columns exist
        if all(col in df.columns for col in feature_cols) and target_col in df.columns:
            
            X = df[feature_cols]
            # Map 'Yes'/'No' to 1/0 if necessary
            if df[target_col].dtype == 'object':
                y = df[target_col].map({'No': 0, 'Yes': 1})
            else:
                y = df[target_col]

            # Data Splitting
            test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.25)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.success(f"Data processed. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

            # --- 3. MODEL TRAINING ---
            st.subheader("Model Performance")
            
            if st.button("Train & Compare Models"):
                with st.spinner("Training models..."):
                    models = {
                        "SVM": SVC(kernel='rbf', probability=True),
                        "KNN": KNeighborsClassifier(n_neighbors=5),
                        "Naive Bayes": GaussianNB(),
                        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }

                    results = []
                    for name, clf in models.items():
                        # Use scaled data for SVM/KNN, raw for Trees (optional but good practice)
                        if name in ["SVM", "KNN"]:
                            clf.fit(X_train_scaled, y_train)
                            metrics = evaluate_model(clf, X_test_scaled, y_test)
                        else:
                            clf.fit(X_train, y_train)
                            metrics = evaluate_model(clf, X_test, y_test)
                        
                        metrics["Model"] = name
                        results.append(metrics)

                    results_df = pd.DataFrame(results).set_index("Model")
                    st.table(results_df.style.highlight_max(axis=0))

                    # Find best model for SHAP
                    best_model_name = results_df["Accuracy"].idxmax()
                    st.info(f"Best performing model based on Accuracy: **{best_model_name}**")

            # --- 4. HYPERPARAMETER TUNING (XGBoost) ---
            st.subheader("Advanced: XGBoost Tuning")
            if st.checkbox("Run Grid Search on XGBoost"):
                with st.spinner("Tuning XGBoost (this takes time)..."):
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 4]
                    }
                    
                    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='roc_auc')
                    grid.fit(X_train, y_train)
                    
                    st.write("Best Parameters:", grid.best_params_)
                    
                    best_xgb = grid.best_estimator_
                    
                    # SHAP Analysis on best model
                    st.markdown("### Feature Importance (SHAP)")
                    explainer = shap.TreeExplainer(best_xgb)
                    shap_values = explainer.shap_values(X_train)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Summary Plot**")
                        fig1, ax1 = plt.subplots()
                        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.markdown("**Beeswarm Plot**")
                        fig2, ax2 = plt.subplots()
                        shap.summary_plot(shap_values, X_train, show=False)
                        st.pyplot(fig2)

        else:
            st.error(f"Dataset must contain these columns: {', '.join(feature_cols)} and '{target_col}'")
else:
    st.info("Awaiting file upload. Please upload your Excel file from the sidebar.")
