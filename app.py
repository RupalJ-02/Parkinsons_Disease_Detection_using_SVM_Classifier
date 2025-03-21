from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form[key]) for key in request.form]
        data_scaled = scaler.transform([data])  # Scale input
        prediction = model.predict(data_scaled)  # Predict

        result = "Parkinson's Detected" if prediction[0] > 0.5 else "Healthy"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
