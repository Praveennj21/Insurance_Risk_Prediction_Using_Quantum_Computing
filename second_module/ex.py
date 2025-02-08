# Load the required models
import joblib
import pandas as pd

# Load your models
health_model = joblib.load('health_model.pkl')
health_svd = joblib.load('health_svd.pkl')
health_pca = joblib.load('health_pca.pkl')
preprocessor = joblib.load('health_preprocessor.pkl')

# Sample input data (this should match the structure expected by your model)
input_data = {
    'age': [30],
    'bmi': [25.5],
    'cholesterol': [1],
    'blood_pressure': [120],
    'annual_income': [50000],
    'annual_claims': [2000],
    'num_doctor_visits': [5],
    'num_specialist_visits': [2],
    'gender': ['Male'],
    'smoking_status': ['Non-smoker'],
    'exercise_frequency': ['Weekly'],
    'alcohol_consumption': ['Occasionally'],
    'pre_existing_conditions': ['None'],
    'marital_status': ['Single'],
    'residential_area': ['Urban'],
    'family_medical_history': ['None'],
    'healthcare_access': ['Good']
}

# Convert input data to DataFrame
preprocessed_input = pd.DataFrame(input_data)

# Preprocess the input data
preprocessed_input = preprocessor.transform(preprocessed_input)

# Transform the preprocessed data using SVD
svd_input = health_svd.transform(preprocessed_input)

# Apply PCA transformation
pca_input = health_pca.transform(svd_input)

# Make prediction
risk_level = health_model.predict(pca_input)[0]

# Output results
print("Predicted Risk Level:", risk_level)
