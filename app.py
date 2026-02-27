from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load Saved Model - Ensure these files exist in your directory
try:
    model = joblib.load("xgb_heart_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except:
    print("Warning: Model files not found. Please ensure .pkl files are in the directory.")

def predict_heart_disease(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HeartHealth AI | Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --danger: #ef4444;
            --success: #22c55e;
            --bg: #0f172a;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background: radial-gradient(circle at top right, #1e293b, #0f172a);
            color: #f8fafc;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 24px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        h2 { font-weight: 600; margin-bottom: 30px; letter-spacing: -1px; }
        
        .form-group { text-align: left; margin-bottom: 18px; }
        
        label { display: block; font-size: 0.85rem; margin-bottom: 6px; color: #94a3b8; }

        input, select {
            width: 100%;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.2);
            color: white;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            width: 100%;
            background: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }

        button:hover { transform: translateY(-2px); opacity: 0.9; }

        /* Result Styling */
        .result-card {
            margin-top: 30px;
            padding: 20px;
            border-radius: 16px;
            background: rgba(0, 0, 0, 0.2);
            animation: fadeIn 0.5s ease-out;
        }

        .risk-bar-bg {
            background: #334155;
            height: 8px;
            border-radius: 10px;
            margin: 15px 0;
            overflow: hidden;
        }

        .risk-bar-fill {
            height: 100%;
            transition: width 1s ease-in-out;
        }

        .high-risk { color: var(--danger); }
        .low-risk { color: var(--success); }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h2>❤️ HeartHealth <span style="color:var(--primary)">AI</span></h2>
    
    <form method="POST">
        <div class="form-group">
            <label>Sex</label>
            <select name="sex">
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label>Chest Pain Type</label>
            <select name="cp">
                <option value="1">Typical Angina</option>
                <option value="2">Atypical Angina</option>
                <option value="3">Non-anginal Pain</option>
                <option value="4">Asymptomatic</option>
            </select>
        </div>

        <div class="form-group">
            <label>Cholesterol (mg/dL)</label>
            <input type="number" name="chol" placeholder="e.g. 200" required>
        </div>

        <div class="form-group">
            <label>Resting ECG</label>
            <select name="ecg">
                <option value="0">Normal</option>
                <option value="1">ST-T Wave Abnormality</option>
                <option value="2">Left Ventricular Hypertrophy</option>
            </select>
        </div>

        <div class="form-group">
            <label>Exercise Induced Angina</label>
            <select name="angina">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>

        <div style="display: flex; gap: 10px;">
            <div class="form-group" style="flex:1">
                <label>Oldpeak</label>
                <input type="number" step="0.1" name="oldpeak" placeholder="0.0" required>
            </div>
            <div class="form-group" style="flex:1">
                <label>ST Slope</label>
                <select name="slope">
                    <option value="1">Upsloping</option>
                    <option value="2">Flat</option>
                    <option value="3">Downsloping</option>
                </select>
            </div>
        </div>

        <button type="submit">Analyze Health Data</button>
    </form>

    {% if result %}
    <div class="result-card">
        <h3 class="{{ 'high-risk' if 'Detected' in result else 'low-risk' }}">
            {{ result }}
        </h3>
        <p style="font-size: 0.9rem; color: #94a3b8;">Risk Probability: {{ prob }}%</p>
        <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width: {{ prob }}%; background: {{ '#ef4444' if prob > 50 else '#22c55e' }};"></div>
        </div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    prob = None

    if request.method == "POST":
        try:
            patient = {
                "sex": int(request.form["sex"]),
                "chest pain type": int(request.form["cp"]),
                "cholesterol": float(request.form["chol"]),
                "resting ecg": int(request.form["ecg"]),
                "exercise angina": int(request.form["angina"]),
                "oldpeak": float(request.form["oldpeak"]),
                "ST slope": int(request.form["slope"]),
            }

            pred, probability = predict_heart_disease(patient)
            result = "⚠️ Heart Disease Detected" if pred == 1 else "✅ No Heart Disease"
            prob = round(probability * 100, 2)
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template_string(HTML_PAGE, result=result, prob=prob)

if __name__ == "__main__":
    app.run(debug=True)
