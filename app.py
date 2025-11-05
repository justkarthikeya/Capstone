import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# --- Global Configurations (Colors and Artifacts) ---
colors = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Magenta
    'accent1': '#F18F01',      # Orange
    'accent2': '#C73E1D',      # Red (Churn)
    'accent3': '#3BBA9C',      # Teal (Stay)
    'light': '#F0F0F0',        # Light gray
    'dark': '#2B2D42'          # Dark blue
}

# Custom colormap for visualizations
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [colors['accent3'], colors['accent1'], colors['accent2']])

# --- Load Model and Preprocessing Artifacts ---
# NOTE: Removed @st.cache_resource, standard global loading is used for Gradio.
try:
    # Model and preprocessing artifacts were created using scikit-learn v1.6.1 in the notebook.
    # We load them here, but the environment must have matching library versions (e.g., scikit-learn and category_encoders)
    model = joblib.load('churn_model.pkl')
    num_imputer = joblib.load('num_imputer.pkl')
    num_scaler = joblib.load('num_scaler.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    num_cols = joblib.load('num_cols.pkl')
    cat_cols = joblib.load('cat_cols.pkl')
    
    # Define all columns for SHAP plotting
    all_cols = num_cols + cat_cols
    
except FileNotFoundError:
    print("FATAL ERROR: Model or preprocessing artifacts not found. Check file paths and existence.")
    model, num_imputer, num_scaler, target_encoder, num_cols, cat_cols = None, None, None, None, [], []
    all_cols = []
except ImportError as e:
    print(f"FATAL ERROR: A required library is missing: {e}. Ensure 'requirements.txt' is correct.")
    # Fallback to None if import fails (e.g., category_encoders missing)
    model, num_imputer, num_scaler, target_encoder, num_cols, cat_cols = None, None, None, None, [], []
    all_cols = []


# --- Helper Functions (From Streamlit App) ---
def add_features(df_in):
    X = df_in.copy()
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    # Ensure all service columns exist before calling .sum(axis=1)
    for col in service_cols:
        if col not in X.columns:
            # Assume 'No' if the service column is missing in a simplified single-row input
            X[col] = 'No' 
    
    # Feature Engineering (as defined in Block 5 of the notebook)
    X['ServicesCount'] = (X[service_cols] == 'Yes').sum(axis=1)
    X['ContractOrd'] = X['Contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2})
    X['IsAutoPay'] = X['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    X['SimpleCLV'] = X['MonthlyCharges'] * X['tenure']
    X['HasInternet'] = (X['InternetService'] != 'No').astype(int)
    
    return X

# Load Data AND Apply All Cleaning/Feature Engineering
try:
    # Ensure the data file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is present
    df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    # Use the cleaning logic from the notebook (Block 3)
    df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median(), inplace=True) #
    df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].map({1: 'Yes', 0: 'No'}) #
    df_raw['Churn'] = df_raw['Churn'].map({'Yes': 1, 'No': 0}) #
    
    df_fe = add_features(df_raw.drop(columns=['customerID', 'Churn']))
    
    # Initialize SHAP Explainer (only if model and data loading succeeded)
    background_numeric = num_scaler.transform(num_imputer.transform(df_fe[num_cols]))
    background_categorical = target_encoder.transform(df_fe[cat_cols])
    background_data = np.hstack([background_numeric, background_categorical])
    # Explainer uses predict_proba for Logistic Regression (best_model in notebook)
    explainer = shap.Explainer(model.predict_proba, background_data, feature_names=all_cols) 

except FileNotFoundError:
    print("WARNING: Data file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. EDA and SHAP initialization will not work.")
    df_raw, df_fe = pd.DataFrame(), pd.DataFrame()
    explainer = None
except Exception as e:
    print(f"SHAP Explainer or Preprocessing Init Error: {e}")
    explainer = None
    df_raw, df_fe = pd.DataFrame(), pd.DataFrame()


# --- Prediction Function (Gradio API) ---
def predict_churn(*input_values):
    """
    Takes a tuple of input values from Gradio UI components, preprocesses them,
    makes a prediction, generates a local SHAP plot, and returns results.
    """
    # NOTE: The order must strictly match the component order in the gr.Blocks interface
    feature_names = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'Contract',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'MonthlyCharges', 'TotalCharges', 'PaymentMethod',
        'PaperlessBilling'
    ]
    input_data = dict(zip(feature_names, input_values))

    # Guard against missing artifacts
    if model is None or num_imputer is None or num_scaler is None or target_encoder is None or explainer is None:
        return "Model Not Loaded", "Model Not Loaded", "Model Not Loaded", "Model Not Loaded", "Model Not Loaded", None

    try:
        input_df = pd.DataFrame([input_data])
        input_df = add_features(input_df)

        # Reorder and handle missing columns in input_df (critical for correct preprocessing)
        input_df = input_df.reindex(columns=df_fe.columns, fill_value=np.nan)
        
        # Preprocess
        X_num = num_imputer.transform(input_df[num_cols])
        X_num_scaled = num_scaler.transform(X_num)
        # TargetEncoder expects categorical columns to be objects, not arrays
        X_cat = target_encoder.transform(input_df[cat_cols]) 
        X_final = np.hstack([X_num_scaled, X_cat])

        # Predict (using the probability for the positive class: Churn=1)
        churn_prob = model.predict_proba(X_final)[0][1]

        # --- SHAP Local Explanation ---
        # SHAP explainer was defined using predict_proba, so index 1 is for Churn (positive class)
        shap_values = explainer(X_final)
        shap_values_churn = shap_values.values[0, :, 1]
        
        # Prepare SHAP data for horizontal bar chart
        force_df = pd.DataFrame({
            'feature': all_cols,
            'shap_value': shap_values_churn,
            # Use raw input values for display, not processed (scaled/encoded) values
            'raw_value': [input_df[f].iloc[0] for f in all_cols] 
        })
        force_df['abs_shap'] = np.abs(force_df['shap_value'])
        force_df = force_df.sort_values('abs_shap', ascending=False).head(10).copy()
        
        # Add a column for the label/value to display on the plot
        force_df['display_label'] = force_df.apply(
            lambda row: f"{row['feature']} ({row['raw_value']})", axis=1
        )

        # Create SHAP horizontal bar chart (using Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars: accent2 (Red) for positive SHAP (increases Churn prob), accent3 (Teal) for negative
        bar_colors = [colors['accent2'] if x > 0 else colors['accent3'] for x in force_df['shap_value']]
        
        y_pos = np.arange(len(force_df))
        ax.barh(y_pos, force_df['shap_value'], color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(force_df['display_label'])
        ax.set_xlabel('SHAP Value (Impact on Churn Probability)')
        ax.set_title('Top Factors Influencing This Prediction', fontsize=16, color=colors['primary'])

        # Add value labels
        for i, v in enumerate(force_df['shap_value']):
            ax.text(v + 0.01 if v > 0 else v - 0.05, i, f'{v:.3f}',
                    color='black', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        shap_plot = fig
        plt.close(fig) # Critical: Close plot to free memory

        # Determine risk and action
        risk = "🔴 High" if churn_prob > 0.7 else "🟡 Medium" if churn_prob > 0.4 else "🟢 Low"
        action = "Immediate Retention Offer" if churn_prob > 0.7 else "Targeted Offer" if churn_prob > 0.4 else "Monitor Account Health"
        
        # Format probability for HTML display (to replicate Streamlit's progress bar)
        prob_text = f"{churn_prob*100:.1f}%"
        prob_html = f"""
        <div class="card">
            <h3 style="color: {colors['primary']}; margin-top: 0;">Churn Probability</h3>
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {churn_prob*100}%;
                        background: linear-gradient(90deg, {colors['accent3']} 0%, {colors['accent1']} 50%, {colors['accent2']} 100%);">
                    {prob_text}
                </div>
            </div>
            <p style="margin-top: 1rem;">
                This customer has a <b>{churn_prob*100:.1f}%</b> probability of churning.
            </p>
        </div>
        """

        return risk, action, prob_text, prob_html, shap_plot

    except Exception as e:
        print(f"Prediction error: {e}")
        # Return error messages/placeholders for all outputs
        return "Error", "Error", "Error", f"Prediction failed: {e}", None


# --- EDA Plotting Functions (Converted to Gradio-compatible functions) ---
# NOTE: All plotting functions must return a matplotlib figure object and close it
def plot_churn_distribution(df):
    if df.empty: return plt.figure()
    fig, ax = plt.subplots(figsize=(7, 7))
    churn_counts = df['Churn'].value_counts()
    labels = ['Stayed', 'Churned']
    colors_pie = [colors['accent3'], colors['accent2']]
    
    wedges, texts, autotexts = ax.pie(
        churn_counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops={'fontsize': 14}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title("Overall Customer Churn Distribution", fontsize=16, color=colors['primary'], pad=20)
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_churn_by_contract(df):
    if df.empty: return plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    x = np.arange(len(contract_churn.index))
    width = 0.35
    bars1 = ax.bar(x - width/2, contract_churn[0], width, label='Stayed', color=colors['accent3'])
    bars2 = ax.bar(x + width/2, contract_churn[1], width, label='Churned', color=colors['accent2'])
    ax.set_xlabel('Contract Type', fontsize=12)
    ax.set_ylabel('Percentage of Customers', fontsize=12)
    ax.set_title('Churn Rate by Contract Type', fontsize=16, color=colors['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(contract_churn.index)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_tenure_distribution(df):
    if df.empty: return plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    stayed = df[df['Churn'] == 0]['tenure']
    churned = df[df['Churn'] == 1]['tenure']
    ax.hist([stayed, churned], bins=30, stacked=True,
            color=[colors['accent3'], colors['accent2']],
            label=['Stayed', 'Churned'], alpha=0.7)
    ax.set_xlabel('Tenure (Months)', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Churn Rate by Customer Tenure', fontsize=16, color=colors['primary'])
    ax.legend()
    plt.tight_layout()
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_monthly_charges_distribution(df):
    if df.empty: return plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    stayed = df[df['Churn'] == 0]['MonthlyCharges']
    churned = df[df['Churn'] == 1]['MonthlyCharges']
    ax.hist([stayed, churned], bins=30, stacked=True,
            color=[colors['accent3'], colors['accent2']],
            label=['Stayed', 'Churned'], alpha=0.7)
    ax.set_xlabel('Monthly Charges ($)', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Churn Rate by Monthly Charges', fontsize=16, color=colors['primary'])
    ax.legend()
    plt.tight_layout()
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_services_heatmap(df):
    if df.empty: return plt.figure()
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    service_data = df[service_cols].applymap(lambda x: 1 if x == 'Yes' else 0)
    service_data['Churn'] = df['Churn']
    corr = service_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap=custom_cmap, vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    ax.set_title('Service Correlation with Churn', fontsize=16, color=colors['primary'], pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_demographic_insights(df):
    if df.empty: return plt.figure()
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    titles = ['Churn by Gender', 'Churn by Senior Citizen Status',
              'Churn by Partner Status', 'Churn by Dependents Status']
    colors_pie = [[colors['primary'], colors['secondary']],
                  [colors['accent1'], colors['accent2']],
                  [colors['accent3'], colors['secondary']],
                  [colors['primary'], colors['accent1']]]
    for i, feature in enumerate(demographic_features):
        ax = fig.add_subplot(gs[i//2, i%2])
        churn_rates = df.groupby(feature)['Churn'].mean() * 100
        wedges, texts, autotexts = ax.pie(
            churn_rates.values,
            labels=churn_rates.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_pie[i],
            wedgeprops=dict(width=0.5, edgecolor='w')
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(titles[i], fontsize=14, color=colors['primary'])
    fig.suptitle('Demographic Insights on Churn', fontsize=16, color=colors['primary'])
    plt.tight_layout()
    plt.close(fig) # Critical: Close plot to free memory
    return fig

def plot_global_shap_summary():
    if explainer is None: return plt.figure()
    try:
        # Calculate SHAP values for a sample of the background data
        # Note: We limit the background data size for performance
        shap_values = explainer(explainer.background_dataset[:200]) 

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plotting for the "Churn" class (index 1)
        shap.summary_plot(shap_values.values[:,:,1],
                          features=explainer.background_dataset[:200],
                          feature_names=all_cols,
                          show=False,
                          plot_type="dot",
                          cmap=custom_cmap)
        plt.title("Global Feature Importance for Churn Prediction", fontsize=16, color=colors['primary'], pad=20)
        plt.tight_layout()
        plt.close(fig) # Critical: Close plot to free memory
        return fig
    except Exception as e:
        print(f"Error generating global SHAP plot: {e}")
        return plt.figure()


# --- Gradio UI Definition using gr.Blocks ---
with gr.Blocks(title="Telco Churn Prediction Hub", theme=gr.themes.Soft(), css="""
    /* Custom CSS for Gradio to emulate Streamlit style */
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }

    /* Header Styling */
    .header-box {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .header-box h1 {
        color: white;
        margin: 0;
        text-align: center;
        font-size: 2.5em;
    }
    .header-box p {
        color: rgba(255,255,255,0.8);
        text-align: center;
        margin: 0.5rem 0 0 0;
    }

    /* Sidebar/Input Column Styling */
    .sidebar-box {
        background: linear-gradient(180deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        height: 100%; /* Make it fill the column */
        overflow-y: auto;
    }
    .sidebar-box h2 {
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Card Styling (for the custom HTML output) */
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #2E86AB;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Progress Bar (simplified HTML/CSS injection via gr.HTML) */
    .progress-bar-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 10px;
        height: 30px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    .progress-bar {
        height: 100%;
        line-height: 30px;
        text-align: center;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        transition: width 0.5s ease-in-out, background-color 0.5s ease-in-out;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
""") as demo:

    # Global Header
    gr.HTML(f"""
    <div class="header-box">
        <h1>📡 Telco Churn Prediction Hub</h1>
        <p>Predict customer churn and identify key drivers with this advanced analytics platform</p>
    </div>
    """)

    # --- UI Components Definition ---
    # Define features in the exact order the 'predict_churn' function expects them
    gender_input = gr.Dropdown(df_raw['gender'].unique().tolist() if not df_raw.empty else ['Male', 'Female'], label="Gender", value='Female')
    senior_citizen_input = gr.Dropdown(['No', 'Yes'], label="Senior Citizen", value='No')
    partner_input = gr.Dropdown(['No', 'Yes'], label="Partner", value='Yes')
    dependents_input = gr.Dropdown(['No', 'Yes'], label="Dependents", value='No')
    tenure_input = gr.Slider(minimum=1, maximum=72, value=1, label="Tenure (Months)", step=1)
    contract_input = gr.Dropdown(df_raw['Contract'].unique().tolist() if not df_raw.empty else ['Month-to-month', 'One year', 'Two year'], label="Contract", value='Month-to-month')
    phone_service_input = gr.Dropdown(['No', 'Yes'], label="Phone Service", value='Yes')
    multiple_lines_input = gr.Dropdown(df_raw['MultipleLines'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No phone service'], label="Multiple Lines", value='No')
    internet_service_input = gr.Dropdown(df_raw['InternetService'].unique().tolist() if not df_raw.empty else ['DSL', 'Fiber optic', 'No'], label="Internet Service", value='DSL')
    online_security_input = gr.Dropdown(df_raw['OnlineSecurity'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Online Security", value='No')
    online_backup_input = gr.Dropdown(df_raw['OnlineBackup'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Online Backup", value='Yes')
    device_protection_input = gr.Dropdown(df_raw['DeviceProtection'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Device Protection", value='No')
    tech_support_input = gr.Dropdown(df_raw['TechSupport'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Tech Support", value='No')
    streaming_tv_input = gr.Dropdown(df_raw['StreamingTV'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Streaming TV", value='No')
    streaming_movies_input = gr.Dropdown(df_raw['StreamingMovies'].unique().tolist() if not df_raw.empty else ['No', 'Yes', 'No internet service'], label="Streaming Movies", value='No')
    monthly_charges_input = gr.Slider(minimum=18.0, maximum=120.0, value=29.85, label="Monthly Charges ($)")
    total_charges_input = gr.Slider(minimum=18.0, maximum=9000.0, value=29.85, label="Total Charges ($)")
    payment_method_input = gr.Dropdown(df_raw['PaymentMethod'].unique().tolist() if not df_raw.empty else ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label="Payment Method", value='Electronic check')
    paperless_billing_input = gr.Dropdown(['No', 'Yes'], label="Paperless Billing", value='Yes')
    
    input_components = [
        gender_input, senior_citizen_input, partner_input, dependents_input,
        tenure_input, contract_input, phone_service_input, multiple_lines_input,
        internet_service_input, online_security_input, online_backup_input,
        device_protection_input, tech_support_input, streaming_tv_input,
        streaming_movies_input, monthly_charges_input, total_charges_input,
        payment_method_input, paperless_billing_input
    ]

    with gr.Row():
        # --- Sidebar/Input Column (Left) ---
        with gr.Column(scale=1):
            # FIX: Replaced gr.Box with gr.Group for component containment
            with gr.Group(elem_classes="sidebar-box"): 
                gr.Markdown("## 👤 Customer Details")

                with gr.Accordion(label="📋 Customer Information", open=True):
                    gr.Row(
                        gender_input,
                        senior_citizen_input
                    )
                    gr.Row(
                        partner_input,
                        dependents_input
                    )
                    
                with gr.Accordion(label="🌐 Account & Services", open=False):
                    gr.Row(
                        tenure_input,
                        contract_input
                    )
                    
                    gr.Row(
                        phone_service_input,
                        multiple_lines_input
                    )
                    
                    gr.Row(
                        internet_service_input
                    )
                    
                    gr.Row(
                        online_security_input,
                        online_backup_input,
                        device_protection_input
                    )

                    gr.Row(
                        tech_support_input,
                        streaming_tv_input,
                        streaming_movies_input
                    )

                with gr.Accordion(label="💳 Billing & Payment", open=False):
                    gr.Row(
                        monthly_charges_input,
                        total_charges_input
                    )
                    gr.Row(
                        payment_method_input,
                        paperless_billing_input
                    )
                    

                predict_button = gr.Button("🔮 Predict Churn Probability", variant="primary", scale=0)

        # --- Main Content Column (Right) ---
        with gr.Column(scale=3):
            with gr.Tab("📊 Prediction Results"):
                gr.Markdown("## 📊 Prediction Summary")

                # Outputs for prediction results
                risk_output = gr.Markdown(label="Retention Priority", value="<div class='metric-value'>Click Predict</div>")
                action_output = gr.Markdown(label="Recommended Action", value="<div class='metric-value'>---</div>")
                churn_prob_text = gr.Markdown(label="Probability Value", value="<div class='metric-value'>---</div>")
                
                # HTML output for the custom progress bar
                churn_prob_html = gr.HTML(label="Visual Probability", elem_classes="card", value="<div class='card'>Waiting for input...</div>") 
                
                gr.Markdown("## 🔍 Churn Drivers")
                gr.Markdown("The following factors are contributing to this customer's churn probability (Local SHAP Explanation):")
                shap_plot_output = gr.Plot(label="Top Factors Influencing This Prediction")


            with gr.Tab("📈 Exploratory Data Analysis"):
                gr.Markdown("## 📈 Exploratory Data Analysis")

                with gr.Tabs():
                    with gr.TabItem("📊 Overview"):
                        gr.Markdown("### Overall Churn Distribution and Service Correlation")
                        with gr.Row():
                            # Call the function without arguments to plot the pre-loaded data
                            churn_dist_plot = gr.Plot(value=plot_churn_distribution(df_raw)) 
                            gr.Markdown("""
                                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 400px;">
                                    <h4 style="color: #2E86AB; margin-top: 0;">Key Overall Insights</h4>
                                    <ul style="color: #555;">
                                        <li>The dataset has an **overall churn rate of ~26.5%**.</li>
                                        <li>This suggests the model is dealing with a moderate class imbalance.</li>
                                        <li>High churn rates are typically seen in the **Month-to-Month contract segment**.</li>
                                    </ul>
                                </div>
                            """)
                        gr.Plot(value=plot_services_heatmap(df_raw), label="Service Correlation with Churn (Positive correlation = High Churn)")

                    with gr.TabItem("📋 Contract & Billing"):
                        gr.Markdown("### Contract and Monthly Charges Impact")
                        with gr.Row():
                            gr.Plot(value=plot_churn_by_contract(df_raw), label="Churn Rate by Contract Type")
                            gr.Plot(value=plot_monthly_charges_distribution(df_raw), label="Churn Rate by Monthly Charges")
                        gr.Markdown("""
                            <div class="card" style="border-left: 5px solid #A23B72;">
                                <h4>Key Retention Focus Areas</h4>
                                <ul>
                                    <li>**Contract**: Month-to-month contracts have the highest churn risk.</li>
                                    <li>**Charges**: Customers with very low and very high monthly charges tend to churn more often.</li>
                                </ul>
                            </div>
                        """)


                    with gr.TabItem("⏰ Tenure & Demographics"):
                        gr.Markdown("### Customer Tenure and Demographic Insights")
                        with gr.Row():
                            gr.Plot(value=plot_tenure_distribution(df_raw), label="Churn Rate by Customer Tenure")
                            gr.Plot(value=plot_demographic_insights(df_raw), label="Churn Rates by Demographic Features")
                        gr.Markdown("""
                            <div class="card" style="border-left: 5px solid #F18F01;">
                                <h4>Tenure and Demographic Takeaways</h4>
                                <ul>
                                    <li>**Tenure**: New customers (under 12 months) are highly susceptible to churn.</li>
                                    <li>**Senior Citizens**: Senior citizens (marked 'Yes') show a higher churn rate than non-seniors.</li>
                                </ul>
                            </div>
                        """)

            with gr.Tab("🌐 Global Feature Importance"):
                gr.Markdown("## 🌐 Global Feature Importance (Global SHAP Summary)")
                gr.Markdown("""
                This plot shows the top 10 features overall. Each point represents a customer. 
                Color indicates the feature value (Red = High, Blue = Low, based on the Target Encoding/Scaling).
                """)
                gr.Plot(value=plot_global_shap_summary(), label="Global Feature Importance (SHAP Summary Plot)")


    # --- Gradio Event Handling (The 'predict_button' action) ---
    output_components = [
        risk_output, action_output, churn_prob_text, churn_prob_html, shap_plot_output
    ]

    predict_button.click(
        fn=predict_churn,
        inputs=input_components,
        outputs=output_components,
        api_name="predict"
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    demo.launch()