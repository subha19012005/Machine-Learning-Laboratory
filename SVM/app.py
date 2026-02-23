from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load everything
model = joblib.load("svm_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = {
            'gender': request.form['gender'],
            'married': request.form['married'],
            'dependents': request.form['dependents'],
            'education': request.form['education'],
            'self_employed': request.form['self_employed'],
            'applicantincome': float(request.form['applicantincome']),
            'coapplicantincome': float(request.form['coapplicantincome']),
            'loanamount': float(request.form['loanamount']),
            'loan_amount_term': float(request.form['loan_amount_term']),
            'credit_history': float(request.form['credit_history']),
            'property_area': request.form['property_area']
        }

        sample_df = pd.DataFrame([data])

        # Convert categorical to lowercase (IMPORTANT)
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                sample_df[col] = sample_df[col].str.lower()

        # Encode categorical columns
        for col in sample_df.columns:
            if col in encoders:
                sample_df[col] = encoders[col].transform(sample_df[col])

        # Apply imputer
        sample_df = imputer.transform(sample_df)

        # Apply scaler
        sample_df = scaler.transform(sample_df)

        # Predict
        prediction = model.predict(sample_df)
        result = encoders['loan_status'].inverse_transform(prediction)

        return render_template(
            'index.html',
            prediction=result[0]
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction="Error: Invalid Input"
        )

if __name__ == '__main__':
    app.run(debug=True)