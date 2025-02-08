from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

# Load the trained model
model = joblib.load('quantum_kernel_random_forest_classifier.pkl')

# Define numeric and categorical features
numeric_features = [
    'age', 'annual_income', 'vehicle_age', 'engine_size', 
    'mileage_driven_annually', 'accident_history', 'traffic_violations', 
    'claims_history', 'license_duration'
]
categorical_features = ['gender', 'residential_location', 'vehicle_make']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit transformers and save the preprocessor
def fit_and_save_preprocessor(training_data_path):
    training_data = pd.read_csv(training_data_path)
    preprocessed_training_data = preprocessor.fit_transform(training_data)
    
    # Save the fitted preprocessor
    joblib.dump(preprocessor, 'fitted_preprocessor.pkl')

# Initialize Flask app
app = Flask(__name__)

# Call this function to fit and save the preprocessor once
fit_and_save_preprocessor('precise_insurance_data2.csv')

@app.route('/', methods=['GET', 'POST'])
def collect_data():
    if request.method == 'POST':
        data = {
            'age': float(request.form.get('age', 0)),
            'annual_income': float(request.form.get('annual_income', 0)),
            'vehicle_age': float(request.form.get('vehicle_age', 0)),
            'engine_size': float(request.form.get('engine_size', 0)),
            'mileage_driven_annually': float(request.form.get('mileage_driven_annually', 0)),
            'accident_history': float(request.form.get('accident_history', 0)),
            'traffic_violations': float(request.form.get('traffic_violations', 0)),
            'claims_history': float(request.form.get('claims_history', 0)),
            'license_duration': float(request.form.get('license_duration', 0)),
            'gender': request.form.get('gender', ''),
            'residential_location': request.form.get('residential_location', ''),
            'vehicle_make': request.form.get('vehicle_make', '')
        }

        df = pd.DataFrame([data])

        preprocessor = joblib.load('fitted_preprocessor.pkl')
        data_preprocessed = preprocessor.transform(df)

        svd = joblib.load('D:\\second_module\\svd_transformer.pkl')
        kernel_vector = svd.transform(data_preprocessed)

        # Predict the risk level
        prediction = model.predict(kernel_vector)
        
        # Log prediction for debugging
        print(f"Input data: {data}")
        print(f"Prediction: {prediction}")

        # Assuming prediction outputs the risk level directly
        risk_level = prediction[0]  # Adjust if needed

        return redirect(url_for('show_result', risk_level=risk_level))

    return render_template('data_collection.html')


@app.route('/result')
def show_result():
    risk_level = request.args.get('risk_level', None)
    return render_template('result.html', risk_level=risk_level)

if __name__ == '__main__':
    app.run(debug=True)