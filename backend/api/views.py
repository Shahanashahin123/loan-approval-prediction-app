from django.http import JsonResponse
import pickle
import os
import numpy as np

# Load model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(base_dir, '..', 'model.pkl'))
scaler_path = os.path.abspath(os.path.join(base_dir, '..', 'scaler.pkl'))

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def predict_loan(request):
    try:
        features = [
            float(request.GET.get('no_of_dependents')),
            float(request.GET.get('education') == 'Graduate'),  # 1 if Graduate, 0 otherwise
            float(request.GET.get('self_employed') == 'Yes'),   # 1 if Yes, 0 otherwise
            float(request.GET.get('income_annum')),
            float(request.GET.get('loan_amount')),
            float(request.GET.get('loan_term')),
            float(request.GET.get('cibil_score')),
            float(request.GET.get('residential_assets_value')),
            float(request.GET.get('commercial_assets_value')),
            float(request.GET.get('luxury_assets_value')),
            float(request.GET.get('bank_asset_value')),
        ]

        # Scale features
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        return JsonResponse({'loan_status': int(prediction)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
