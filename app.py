import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# Page Config
st.set_page_config(page_title="Earnings Manipulator Detector", layout="wide")

# Title
st.title("📊 Earnings Manipulator Detection")
st.markdown("Based on the Beneish M-Score variables using Machine Learning.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# Sidebar for File Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'Earnings Manipulator.xlsx'", type=["xlsx"])

# Helper function for evaluation
def evaluate(model_name, y_true, y_pred, y_prob):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

if uploaded_file:
    df = load_data(uploaded_file)
    
    # --- 2. PREPROCESSING ---
    st.header("Data Overview")
    st.dataframe(df.head())

    # Define features and target based on your notebook logic
    # We use try/except to handle cases where columns might be missing
    try:
        feature_cols = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
        X = df[feature_cols]
        y = df['Manipulator'].map({'No': 0, 'Yes': 1})
        
        st.success("✅ Features and Target extracted successfully.")
        
        # Train/Test Split
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    except KeyError as e:
        st.error(f"Error: The uploaded file is missing required columns: {e}")
        st.stop()

    # --- 3. MODEL TRAINING & COMPARISON ---
    st.header("Model Comparison")
    
    if st.button("Train Baseline Models"):
        with st.spinner("Training models..."):
            results = []

            # 1. SVM
            svm = SVC(kernel='rbf', probability=True)
            svm.fit(X_train_scaled, y_train)
            results.append(evaluate("SVM", y_test, svm.predict(X_test_scaled), svm.predict_proba(X_test_scaled)[:,1]))

            # 2. KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            results.append(evaluate("KNN", y_test, knn.predict(X_test_scaled), knn.predict_proba(X_test_scaled)[:,1]))

            # 3. Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train) # NB usually doesn't need scaling, but works with it too
            results.append(evaluate("Naive Bayes", y_test, nb.predict(X_test), nb.predict_proba(X_test)[:,1]))

            # 4. AdaBoost
            ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
            ada.fit(X_train, y_train)
            results.append(evaluate("AdaBoost", y_test, ada.predict(X_test), ada.predict_proba(X_test)[:,1]))

            # 5. XGBoost
            xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, 
                                subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', random_state=42)
            xgb.fit(X_train, y_train)
            results.append(evaluate("XGBoost", y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:,1]))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Simple bar chart
            st.bar_chart(results_df.set_index("Model")["Accuracy"])

    # --- 4. HYPERPARAMETER TUNING ---
    st.header("Hyperparameter Tuning (XGBoost)")
    st.info("We will tune XGBoost as it was one of the top performers in the analysis.")
    
    if st.button("Run GridSearch for XGBoost"):
        with st.spinner("Running GridSearch (this may take a moment)..."):
            param_grid_xgb = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4]
            }
            
            xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
            grid_search = GridSearchCV(xgb_base, param_grid_xgb, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_xgb = grid_search.best_estimator_
            
            st.write("**Best Parameters:**", grid_search.best_params_)
            
            # Evaluate best model
            tuned_metrics = evaluate("XGBoost Tuned", y_test, best_xgb.predict(X_test), best_xgb.predict_proba(X_test)[:,1])
            st.table(pd.DataFrame([tuned_metrics]))

            # Save model for SHAP
            st.session_state['best_model'] = best_xgb

    # --- 5. SHAP ANALYSIS ---
    st.header("Explainability (SHAP)")
    
    if 'best_model' in st.session_state:
        model = st.session_state['best_model']
        
        if st.checkbox("Show SHAP Summary Plot"):
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.markdown("### Beeswarm Plot")
                fig2, ax2 = plt.subplots()
                shap.summary_plot(shap_values, X_train, show=False)
                st.pyplot(fig2)
    else:
        st.warning("Please run Hyperparameter Tuning first to generate the best model for SHAP analysis.")

else:
    st.info("👈 Please upload the 'Earnings Manipulator (1).xlsx' file in the sidebar to begin.")
    st.markdown("""
    **Note for Deployment:** Since this app relies on a specific Excel file, you must either:
    1. Upload it using the sidebar once deployed.
    2. Or, add the file to your GitHub repository and change `st.sidebar.file_uploader` to `pd.read_excel('filename.xlsx')` if you want it hardcoded.
    """)
