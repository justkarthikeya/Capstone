import os
import io
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI crashes on macOS when running in a web server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Load artifacts (best-effort). If missing, set to None and handle later.
def _load_artifacts():
    artifacts = {}
    try:
        artifacts['model'] = joblib.load('churn_model.pkl')
        artifacts['num_imputer'] = joblib.load('num_imputer.pkl')
        artifacts['num_scaler'] = joblib.load('num_scaler.pkl')
        artifacts['target_encoder'] = joblib.load('target_encoder.pkl')
        artifacts['num_cols'] = joblib.load('num_cols.pkl')
        artifacts['cat_cols'] = joblib.load('cat_cols.pkl')
        artifacts['all_cols'] = artifacts['num_cols'] + artifacts['cat_cols']
    except Exception:
        # If anything fails, return empty dict keys to avoid crashes.
        artifacts = {'model': None, 'num_imputer': None, 'num_scaler': None, 'target_encoder': None, 'num_cols': [], 'cat_cols': [], 'all_cols': []}

    # Try to create a SHAP explainer if possible and dataset exists
    try:
        if artifacts['model'] is not None and os.path.exists('WA_Fn-UseC_-Telco-Customer-Churn.csv'):
            df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
            df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
            df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median(), inplace=True)
            df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
            # build features
            df_fe = add_features(df_raw.drop(columns=['customerID'], errors='ignore'))
            if artifacts['num_cols'] and artifacts['cat_cols']:
                background_numeric = artifacts['num_scaler'].transform(artifacts['num_imputer'].transform(df_fe[artifacts['num_cols']]))
                background_categorical = artifacts['target_encoder'].transform(df_fe[artifacts['cat_cols']])
                background_data = np.hstack([background_numeric, background_categorical])
                artifacts['explainer'] = shap.Explainer(artifacts['model'].predict_proba, background_data, feature_names=artifacts['all_cols'])
            else:
                artifacts['explainer'] = None
    except Exception:
        artifacts['explainer'] = None

    return artifacts


def add_features(df_in):
    X = df_in.copy()
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in service_cols:
        if col not in X.columns:
            X[col] = 'No'

    X['ServicesCount'] = (X[service_cols] == 'Yes').sum(axis=1)
    if 'Contract' in X.columns:
        X['ContractOrd'] = X['Contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2})
    else:
        X['ContractOrd'] = 0
    X['IsAutoPay'] = X.get('PaymentMethod', '').astype(str).str.contains('automatic', case=False, na=False).astype(int)
    X['SimpleCLV'] = X.get('MonthlyCharges', 0).astype(float) * X.get('tenure', 0).astype(float)
    X['HasInternet'] = (X.get('InternetService', '') != 'No').astype(int)
    return X


ARTIFACTS = _load_artifacts()


def _plot_shap(all_cols, shap_values, input_df):
    # Create horizontal bar chart of top 10 features by abs SHAP
    try:
        shap_arr = shap_values.values[0, :, 1]
        df = pd.DataFrame({'feature': all_cols, 'shap_value': shap_arr, 'raw_value': [input_df[f].iloc[0] if f in input_df.columns else '' for f in all_cols]})
        df['abs_shap'] = df['shap_value'].abs()
        df = df.sort_values('abs_shap', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(df))
        colors = ['#C73E1D' if x>0 else '#3BBA9C' for x in df['shap_value']]
        ax.barh(y_pos, df['shap_value'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']} ({r['raw_value']})" for _, r in df.iterrows()])
        ax.invert_yaxis()
        ax.set_xlabel('SHAP value (impact on churn prob)')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return b64
    except Exception as e:
        # If plotting fails for any reason (backend issues, shap shape mismatch), return None
        try:
            plt.close('all')
        except Exception:
            pass
        return None


def predict_from_dict(input_dict):
    # Return a dict with risk/action/prob_text and optional shap_base64
    model = ARTIFACTS.get('model')
    num_imputer = ARTIFACTS.get('num_imputer')
    num_scaler = ARTIFACTS.get('num_scaler')
    target_encoder = ARTIFACTS.get('target_encoder')
    num_cols = ARTIFACTS.get('num_cols', [])
    cat_cols = ARTIFACTS.get('cat_cols', [])
    all_cols = ARTIFACTS.get('all_cols', [])
    explainer = ARTIFACTS.get('explainer')

    if model is None or num_imputer is None or num_scaler is None or target_encoder is None:
        return {'risk': 'Model Not Loaded', 'action': 'Model Not Loaded', 'prob_text': 'Model Not Loaded'}

    # Build input DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = add_features(input_df)

    try:
        X_num = num_imputer.transform(input_df[num_cols])
        X_num_scaled = num_scaler.transform(X_num)
        X_cat = target_encoder.transform(input_df[cat_cols])
        X_final = np.hstack([X_num_scaled, X_cat])

        churn_prob = model.predict_proba(X_final)[0][1]

        risk = "🔴 High" if churn_prob > 0.7 else "🟡 Medium" if churn_prob > 0.4 else "🟢 Low"
        action = "Immediate Retention Offer" if churn_prob > 0.7 else "Targeted Offer" if churn_prob > 0.4 else "Monitor Account Health"
        prob_text = f"{churn_prob*100:.1f}%"
        result = {'risk': risk, 'action': action, 'prob_text': prob_text, 'prob': float(churn_prob), 'prob_pct': int(churn_prob*100)}

        # SHAP local explanation if explainer is available
        if explainer is not None and all_cols:
            shap_values = explainer(X_final)
            b64 = _plot_shap(all_cols, shap_values, input_df)
            result['shap_base64'] = b64

        return result

    except Exception as e:
        return {'risk': 'Error', 'action': 'Error', 'prob_text': f'Prediction failed: {e}'}
