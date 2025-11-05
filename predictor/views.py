from django.shortcuts import render
from .forms import PredictForm
import model_utils
import decimal
import pandas as pd
import os


def index(request):
    result = None
    shap_img = None

    # Attempt to load dataset to populate form choices dynamically (optional)
    df_raw = None
    csv_path = os.path.join(os.getcwd(), 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if os.path.exists(csv_path):
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception:
            df_raw = None

    if request.method == 'POST':
        form = PredictForm(request.POST)
        # If dataset was loaded, refresh choices to keep UI consistent
        if df_raw is not None:
            # update a few fields with actual choices from data
            try:
                form.fields['Contract'].choices = [(v, v) for v in sorted(df_raw['Contract'].dropna().unique())]
                form.fields['PaymentMethod'].choices = [(v, v) for v in sorted(df_raw['PaymentMethod'].dropna().unique())]
                form.fields['InternetService'].choices = [(v, v) for v in sorted(df_raw['InternetService'].dropna().unique())]
            except Exception:
                pass

        if form.is_valid():
            # Convert form.cleaned_data into the input dict expected by model_utils
            input_data = form.cleaned_data.copy()

            # Normalize numeric types (Decimal -> float, etc.) and ensure tenure is int
            if 'tenure' in input_data:
                try:
                    input_data['tenure'] = int(input_data['tenure'])
                except Exception:
                    input_data['tenure'] = int(float(input_data['tenure']))

            for fld in ['MonthlyCharges', 'TotalCharges']:
                if fld in input_data:
                    val = input_data[fld]
                    if isinstance(val, decimal.Decimal):
                        input_data[fld] = float(val)
                    else:
                        try:
                            input_data[fld] = float(val)
                        except Exception:
                            input_data[fld] = 0.0

            # Call the model utility
            result = model_utils.predict_from_dict(input_data)
            # Debug log: print result to server console so it's visible in runserver output
            try:
                print("PREDICT RESULT:", result)
            except Exception:
                pass
            if result and 'shap_base64' in result:
                shap_img = result['shap_base64']
    else:
        form = PredictForm()
        # Populate choices from data if available
        if df_raw is not None:
            try:
                form.fields['Contract'].choices = [(v, v) for v in sorted(df_raw['Contract'].dropna().unique())]
                form.fields['PaymentMethod'].choices = [(v, v) for v in sorted(df_raw['PaymentMethod'].dropna().unique())]
                form.fields['InternetService'].choices = [(v, v) for v in sorted(df_raw['InternetService'].dropna().unique())]
            except Exception:
                pass

    return render(request, 'predictor/index.html', {'form': form, 'result': result, 'shap_img': shap_img})
