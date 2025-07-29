from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Enhanced HTML with modern UI/UX
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 Wildfire Risk Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --fire-primary: #FF6B35;
            --fire-secondary: #F7C59F;
            --fire-accent: #D9534F;
            --fire-dark: #2E282A;
            --fire-light: #FFF8F0;
            --fire-gradient: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: var(--fire-gradient);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--fire-dark);
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 20px;
        }
        
        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            grid-column: 1 / -1;
        }
        
        .header h1 {
            font-family: 'Montserrat', sans-serif;
            font-size: 3.2rem;
            background: linear-gradient(to right, #ff9966, #ff5e62);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header p {
            font-size: 1.2rem;
            color: var(--fire-light);
            max-width: 700px;
            margin: 0 auto;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        
        .card {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
        }
        
        .form-container {
            padding: 40px;
        }
        
        .form-header {
            margin-bottom: 25px;
        }
        
        .form-header h2 {
            font-size: 2.2rem;
            color: var(--fire-accent);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-header p {
            color: #666;
            line-height: 1.6;
        }
        
        .form-group {
            margin-bottom: 22px;
            position: relative;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--fire-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .input-icon {
            width: 24px;
            height: 24px;
            color: var(--fire-primary);
        }
        
        .form-control {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #e1e5eb;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        
        .form-control:focus {
            outline: none;
            border-color: var(--fire-primary);
            box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2);
            background: #fff;
        }
        
        .submit-btn {
            background: var(--fire-gradient);
            color: white;
            border: none;
            padding: 16px 30px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(255, 94, 98, 0.4);
            margin-top: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        
        .submit-btn:hover {
            background: linear-gradient(135deg, #ff8a5c 0%, #ff4b5c 100%);
            box-shadow: 0 6px 20px rgba(255, 94, 98, 0.6);
            transform: translateY(-2px);
        }
        
        .submit-btn:active {
            transform: translateY(1px);
        }
        
        .info-container {
            padding: 40px;
            display: flex;
            flex-direction: column;
        }
        
        .risk-info {
            flex: 1;
        }
        
        .risk-info h3 {
            font-size: 1.8rem;
            color: var(--fire-accent);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .info-card {
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--fire-primary);
            transition: transform 0.3s ease;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
        }
        
        .info-card h4 {
            font-size: 1.1rem;
            margin-bottom: 10px;
            color: var(--fire-dark);
        }
        
        .info-card p {
            color: #666;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        .result-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-top: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border-top: 5px solid var(--fire-accent);
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-container h4 {
            font-size: 1.4rem;
            color: var(--fire-dark);
            margin-bottom: 15px;
        }
        
        .risk-value {
            font-size: 3.5rem;
            font-weight: 800;
            font-family: 'Montserrat', sans-serif;
            background: var(--fire-gradient);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 10px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .risk-message {
            font-size: 1.3rem;
            color: var(--fire-accent);
            margin-top: 10px;
            font-weight: 600;
        }
        
        .footer {
            grid-column: 1 / -1;
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 0.9rem;
            margin-top: 20px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid var(--fire-accent);
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Wildfire Risk Assessment</h1>
            <p>Predict potential fire risks using meteorological data and forest conditions. Enter environmental parameters to assess fire danger levels.</p>
        </div>
        
        <div class="card form-container">
            <div class="form-header">
                <h2><svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>Input Parameters</h2>
                <p>Provide current environmental conditions to calculate fire risk index</p>
            </div>
            
            <form method="POST" id="prediction-form">
                <div class="form-group">
                    <label for="Temperature">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                        Temperature (°C)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="Temperature" name="Temperature" placeholder="E.g. 23.5" required>
                </div>
                
                <div class="form-group">
                    <label for="RH">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4 4 0 003 15z"></path></svg>
                        Relative Humidity (%)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="RH" name="RH" placeholder="E.g. 45.2" required>
                </div>
                
                <div class="form-group">
                    <label for="Ws">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.59 4.59A2 2 0 1111 8H2m10.59 11.41A2 2 0 1014 16H2m15.73-8.27A2.5 2.5 0 1119.5 12H2"></path></svg>
                        Wind Speed (km/h)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="Ws" name="Ws" placeholder="E.g. 5.5" required>
                </div>
                
                <div class="form-group">
                    <label for="Rain">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4 4 0 003 15z"></path></svg>
                        Rainfall (mm)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="Rain" name="Rain" placeholder="E.g. 0.0" required>
                </div>
                
                <div class="form-group">
                    <label for="FFMC">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z"></path></svg>
                        FFMC (Fine Fuel Moisture Code)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="FFMC" name="FFMC" placeholder="E.g. 86.2" required>
                </div>
                
                <div class="form-group">
                    <label for="DMC">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"></path></svg>
                        DMC (Duff Moisture Code)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="DMC" name="DMC" placeholder="E.g. 26.7" required>
                </div>
                
                <div class="form-group">
                    <label for="ISI">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        ISI (Initial Spread Index)
                    </label>
                    <input type="number" step="0.01" class="form-control" id="ISI" name="ISI" placeholder="E.g. 5.3" required>
                </div>
                
                <div class="form-group">
                    <label for="Classes">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                        Fire Status (0 = None, 1 = Active)
                    </label>
                    <input type="number" class="form-control" id="Classes" name="Classes" placeholder="0 or 1" min="0" max="1" required>
                </div>
                
                <div class="form-group">
                    <label for="Region">
                        <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Region (1 = Bejaia, 2 = Sidi-Bel Abbes)
                    </label>
                    <input type="number" class="form-control" id="Region" name="Region" placeholder="1 or 2" min="1" max="2" required>
                </div>
                
                <button type="submit" class="submit-btn">
                    <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    Calculate Fire Risk
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing fire risk parameters...</p>
            </div>
        </div>
        
        <div class="card info-container">
            <div class="risk-info">
                <h3><svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>Fire Risk Information</h3>
                
                <div class="info-grid">
                    <div class="info-card">
                        <h4>Risk Scale</h4>
                        <p>0-20: Low Risk<br>
                        21-40: Moderate<br>
                        41-60: High<br>
                        61-80: Very High<br>
                        81-100: Extreme</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>Key Factors</h4>
                        <p>• Temperature above 30°C<br>
                        • Humidity below 30%<br>
                        • High Wind Speed<br>
                        • Low Rainfall<br>
                        • High FFMC/DMC values</p>
                    </div>
                </div>
                
                <div class="info-card">
                    <h4>Prevention Tips</h4>
                    <p>• Clear dry vegetation near structures<br>
                    • Maintain firebreaks in forests<br>
                    • Monitor fire weather forecasts<br>
                    • Report any signs of fire immediately<br>
                    • Follow local fire restrictions</p>
                </div>
            </div>
            
            {% if result %}
            <div class="result-container">
                <h4>Fire Risk Prediction</h4>
                <div class="risk-value">{{ result }}</div>
                {% if result|float > 60 %}
                <p class="risk-message">⚠️ Extreme Fire Danger - Take Immediate Precautions</p>
                {% elif result|float > 40 %}
                <p class="risk-message">🔥 High Fire Danger - Be Vigilant</p>
                {% elif result|float > 20 %}
                <p class="risk-message">⚠️ Moderate Fire Risk - Stay Alert</p>
                {% else %}
                <p class="risk-message">✅ Low Fire Risk - Normal Conditions</p>
                {% endif %}
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>Wildfire Prediction System © 2023 | Using Machine Learning for Fire Prevention</p>
        </div>
    </div>
    
    <script>
        document.getElementById('prediction-form').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
        
        // Set sample values for testing
        function setSampleValues() {
            document.getElementById('Temperature').value = 32.5;
            document.getElementById('RH').value = 25.8;
            document.getElementById('Ws').value = 18.3;
            document.getElementById('Rain').value = 0.0;
            document.getElementById('FFMC').value = 92.7;
            document.getElementById('DMC').value = 85.4;
            document.getElementById('ISI').value = 12.6;
            document.getElementById('Classes').value = 1;
            document.getElementById('Region').value = 1;
        }
        
        // Uncomment next line to enable sample values for demo
        // window.onload = setSampleValues;
    </script>
</body>
</html>
'''

@app.route('/', methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            data = [
                float(request.form['Temperature']),
                float(request.form['RH']),
                float(request.form['Ws']),
                float(request.form['Rain']),
                float(request.form['FFMC']),
                float(request.form['DMC']),
                float(request.form['ISI']),
                float(request.form['Classes']),
                float(request.form['Region'])
            ]
            scaled_data = standard_scaler.transform([data])
            prediction = ridge_model.predict(scaled_data)
            result = round(prediction[0], 2)
        except Exception as e:
            result = f"Error: {e}"
    return render_template_string(HTML_PAGE, result=result)

if __name__ == '__main__':
    app.run(debug=True)