
from flask import Flask, send_from_directory, render_template_string, request, abort, jsonify, send_file, Response
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib, numpy as np, io, base64, traceback, matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "stacking_model.pkl"
INDEX_HTML = BASE_DIR / "index.html"
DASHBOARD_HTML = BASE_DIR / "dashboard.html"
BULK_PREDICTION_HTML = BASE_DIR / "bulk_prediction.html"
XTEST_PATH = BASE_DIR / "models" / "x_test.npy"
YTEST_PATH = BASE_DIR / "models" / "y_test.npy"
DATASET_PATH = PARENT_DIR / "Dataset.csv"

app = Flask(__name__, static_folder=str(BASE_DIR))

# Feature names matching training data (prevents warnings and improves performance)
FEATURE_NAMES = [
    'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'AvgCharges'
]

# load model (diagnostic)
model = None
if MODEL_PATH.exists():
    try:
        print("Loading model:", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        model = None
        print("Could not load model:", repr(e))
        traceback.print_exc()
else:
    print("Model file not found:", MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    # GET -> serve index.html
    if request.method == "GET":
        if INDEX_HTML.exists():
            return send_from_directory(str(BASE_DIR), "index.html")
        return render_template_string("<h3>index.html not found</h3><p>Place index.html in the app folder.</p>")

    # POST -> process form from index.html and predict using stacking model
    # Expecting form fields q1 .. q17 as in index.html
    try:
        # collect q1..q15 as ints, q16 total_charges, q17 tenure
        features = []
        for i in range(1, 16):
            val = request.form.get(f"q{i}", None)
            if val is None or val == "":
                return render_template_string(f"<h3>Missing field q{i}</h3><p><a href='/'>Back</a></p>"), 400
            features.append(int(val))
        # q16 total charges (may be float)
        q16 = request.form.get("q16", "0")
        q17 = request.form.get("q17", "0")
        try:
            total_charges = float(q16)
        except:
            total_charges = 0.0
        try:
            tenure = int(float(q17))
        except:
            tenure = 0
        # compute AvgCharges similar to notebook: TotalCharges / tenure (fallback to monthly if tenure 0)
        if tenure > 0:
            avg_charges = total_charges / tenure
        else:
            avg_charges = total_charges
        # append AvgCharges as final feature
        features.append(avg_charges)

        # Create DataFrame with feature names to match training data (prevents warnings)
        X = pd.DataFrame([features], columns=FEATURE_NAMES)

        if model is None:
            return render_template_string(
                "<h3>Model not loaded.</h3>"
                "<p>Place a valid joblib model at <code>Web/models/stacking_model.pkl</code>.</p>"
                "<p><a href='/'>Back</a></p>"
            ), 500

        # Predict
        try:
            pred = model.predict(X)
        except Exception as e:
            return render_template_string(f"<h3>Error during prediction: {e}</h3><p><a href='/'>Back</a></p>"), 500

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0].tolist()
            except Exception:
                proba = None

        # base estimators' predictions (if stacking)
        base_preds = []
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                try:
                    base_preds.append(str(est.predict(X)[0]))
                except Exception:
                    base_preds.append("err")

        # render simple results page with links to graph/index
        html = f"""
        <html>
          <head><title>Prediction Result</title></head>
          <body>
            <h2>Prediction Result</h2>
            <p><strong>Predicted label:</strong> {pred[0]}</p>
            <p><strong>Predicted probabilities:</strong> {proba}</p>
            <p><strong>Base estimators predictions:</strong> {base_preds}</p>
            <p><a href="/">Back to form</a> — <a href="/graph">Open graph.html</a> — <a href="/stacking">Confusion matrix</a></p>
          </body>
        </html>
        """
        return render_template_string(html)

    except Exception as e:
        traceback.print_exc()
        return render_template_string(f"<h3>Server error: {e}</h3><p><a href='/'>Back</a></p>"), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for prediction that returns JSON"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # collect q1..q15 as ints, q16 total_charges, q17 tenure
        features = []
        for i in range(1, 16):
            val = data.get(f"q{i}", None)
            if val is None or val == "":
                return jsonify({"success": False, "error": f"Missing field q{i}"}), 400
            features.append(int(val))
        
        # q16 total charges (may be float)
        q16 = data.get("q16", "0")
        q17 = data.get("q17", "0")
        try:
            total_charges = float(q16)
        except:
            total_charges = 0.0
        try:
            tenure = int(float(q17))
        except:
            tenure = 0
        
        # compute AvgCharges similar to notebook: TotalCharges / tenure (fallback to monthly if tenure 0)
        if tenure > 0:
            avg_charges = total_charges / tenure
        else:
            avg_charges = total_charges
        # append AvgCharges as final feature
        features.append(avg_charges)

        # Create DataFrame with feature names to match training data (prevents warnings)
        X = pd.DataFrame([features], columns=FEATURE_NAMES)

        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        # Predict
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            return jsonify({"success": False, "error": f"Error during prediction: {str(e)}"}), 500

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0].tolist()
            except Exception:
                proba = None

        return jsonify({
            "success": True,
            "prediction": int(pred),
            "prediction_label": "Churn" if pred == 1 else "No Churn",
            "probability": proba if proba else [0.5, 0.5]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    """Serve the dashboard HTML page"""
    if DASHBOARD_HTML.exists():
        return send_from_directory(str(BASE_DIR), "dashboard.html")
    return render_template_string("<h3>dashboard.html not found</h3><p>Dashboard page not available.</p>"), 404



@app.route("/bulk-prediction")
def bulk_prediction():
    """Serve the bulk prediction HTML page"""
    if BULK_PREDICTION_HTML.exists():
        return send_from_directory(str(BASE_DIR), "bulk_prediction.html")
    return render_template_string("<h3>bulk_prediction.html not found</h3>"), 404

@app.route("/api/bulk-predict", methods=["POST"])
def api_bulk_predict():
    """API endpoint for bulk prediction from CSV data"""
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Please upload a valid CSV file"}), 400

        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"success": False, "error": f"Error reading CSV: {str(e)}"}), 400

        # Validate required columns
        required_cols = [
            'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'tenure'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({
                "success": False, 
                "error": f"Missing columns: {', '.join(missing_cols)}"
            }), 400

        # Process features
        try:
            processed_df = process_dataframe(df)
            predictions = make_predictions(processed_df)
            
            return jsonify({
                "success": True,
                "predictions": predictions,
                "total_customers": len(predictions),
                "churn_count": sum(1 for p in predictions if p['prediction'] == 1),
                "no_churn_count": sum(1 for p in predictions if p['prediction'] == 0)
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def process_dataframe(df):
    """Process and encode features"""
    processed_df = df.copy()
    
    # Binary encodings
    binary_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        processed_df[col] = processed_df[col].astype(str).str.lower().apply(
            lambda x: 1 if x in ['yes', '1', 'true'] else 0
        )
    
    # Service encodings
    service_columns = {
        'MultipleLines': _encode_multiple_lines,
        'InternetService': _encode_internet_service,
        'OnlineSecurity': _encode_service,
        'OnlineBackup': _encode_service,
        'DeviceProtection': _encode_service,
        'TechSupport': _encode_service,
        'StreamingTV': _encode_service,
        'StreamingMovies': _encode_service
    }
    
    for col, encoder in service_columns.items():
        processed_df[col] = processed_df[col].apply(encoder)
    
    # Special encodings
    processed_df['Contract'] = processed_df['Contract'].apply(_encode_contract)
    processed_df['PaymentMethod'] = processed_df['PaymentMethod'].apply(_encode_payment_method)
    
    # Numeric processing
    processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce').fillna(0)
    processed_df['tenure'] = pd.to_numeric(processed_df['tenure'], errors='coerce').fillna(0).astype(int)
    
    # Calculate average charges
    processed_df['AvgCharges'] = processed_df['TotalCharges'] / processed_df['tenure'].replace(0, 1)
    processed_df.loc[processed_df['tenure'] == 0, 'AvgCharges'] = processed_df.loc[processed_df['tenure'] == 0, 'TotalCharges']
    
    return processed_df

def make_predictions(processed_df):
    """Make predictions using the model"""
    X = processed_df[FEATURE_NAMES].copy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    predictions_list = []
    for idx, row in processed_df.iterrows():
        pred = int(predictions[idx])
        prob = float(probabilities[idx][1]) if probabilities is not None else 0.5
        
        predictions_list.append({
            'customerID': str(row.get('customerID', f'Customer_{idx+1}')),
            'tenure': int(row['tenure']),
            'MonthlyCharges': float(row.get('MonthlyCharges', row['AvgCharges'])),
            'prediction': pred,
            'churn_probability': prob,
            'top_reason': determine_top_reason({
                name: row[name] for name in FEATURE_NAMES
            }, prob)
        })
    
    return predictions_list

# Keep existing encoding functions unchanged
def _encode_multiple_lines(value):
    """Encode MultipleLines: Yes=2, No=0, No phone service=1"""
    val = str(value).lower()
    if 'yes' in val: return 2
    elif 'no phone' in val or 'no phone service' in val: return 1
    return 0

def _encode_internet_service(value):
    """Encode InternetService: Fiber optic=1, DSL=0, No=2"""
    val = str(value).lower()
    if 'fiber' in val: return 1
    elif 'dsl' in val: return 0
    return 2

def _encode_service(value):
    """Encode services: Yes=2, No=0, No internet service=1"""
    val = str(value).lower()
    if 'yes' in val: return 2
    elif 'no internet' in val or 'no internet service' in val: return 1
    return 0

def _encode_contract(value):
    """Encode Contract: Month-to-month=0, One year=1, Two year=2"""
    val = str(value).lower()
    if 'month' in val: return 0
    elif 'one' in val or '1' in val: return 1
    elif 'two' in val or '2' in val: return 2
    return 0

def _encode_payment_method(value):
    """Encode PaymentMethod: Electronic check=2, Mailed check=3, Bank transfer=0, Credit card=1"""
    val = str(value).lower()
    if 'electronic' in val: return 2
    elif 'mailed' in val: return 3
    elif 'bank' in val: return 0
    elif 'credit' in val: return 1
    return 2

def determine_top_reason(feature_mapping, churn_prob):
    """Determine top reason for churn prediction"""
    reasons = []
    if feature_mapping.get('Contract', 0) == 0:
        reasons.append("Month-to-month contract")
    if churn_prob > 0.7:
        reasons.append("High churn probability")
    if feature_mapping.get('OnlineSecurity', 0) == 0:
        reasons.append("No online security")
    if feature_mapping.get('TechSupport', 0) == 0:
        reasons.append("No tech support")
    return reasons[0] if reasons else "Multiple factors"

@app.route("/api/download-template")
def download_template():
    """Generate and download CSV template"""
    try:
        # Create sample CSV template
        template_data = {
            'customerID': ['CUSTOMER_001', 'CUSTOMER_002'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['Yes', 'Yes'],
            'MultipleLines': ['Yes', 'No'],
            'InternetService': ['Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'Yes'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'Yes'],
            'TechSupport': ['No', 'Yes'],
            'StreamingTV': ['Yes', 'No'],
            'StreamingMovies': ['Yes', 'No'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Bank transfer (automatic)'],
            'MonthlyCharges': [79.85, 53.85],
            'TotalCharges': [3320.75, 1904.75],
            'tenure': [42, 36]
        }
        
        df = pd.DataFrame(template_data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=sample_template.csv'}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# static files fallback
@app.route("/<path:fname>")
def static_file(fname):
    fpath = BASE_DIR / fname
    if fpath.exists() and fpath.is_file():
        return send_from_directory(str(BASE_DIR), fname)
    abort(404)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=500, debug=True)
# ...existing code...