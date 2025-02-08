import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier

# Define numeric and categorical features
numeric_features = [
    'age', 'bmi', 'cholesterol', 'blood_pressure', 
    'annual_income', 'annual_claims', 'num_doctor_visits', 
    'num_specialist_visits'
]
categorical_features = [
    'gender', 'smoking_status', 'exercise_frequency', 
    'alcohol_consumption', 'pre_existing_conditions', 
    'marital_status', 'residential_area', 
    'family_medical_history', 'healthcare_access'
]

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

# Load the health insurance data
training_data = pd.read_csv('pattern_based_health_insurance_risk_data.csv')

# Preprocess the training data
preprocessed_data = preprocessor.fit_transform(training_data)

# Apply Truncated SVD for dimensionality reduction
n_components_svd = min(240, preprocessed_data.shape[1])  # Adjust to feature count
svd = TruncatedSVD(n_components=n_components_svd)
svd_transformed_data = svd.fit_transform(preprocessed_data)

# Apply PCA to further reduce dimensions
n_components_pca = min(32, svd_transformed_data.shape[1])  # Smaller for RandomForest
pca = PCA(n_components=n_components_pca)
pca_transformed_data = pca.fit_transform(svd_transformed_data)

# Train the RandomForestClassifier
target = training_data['risk_level']
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(pca_transformed_data, target)

# Save the RandomForest model and transformers
joblib.dump(rf_model, 'health_model.pkl')
joblib.dump(svd, 'health_svd.pkl')
joblib.dump(pca, 'health_pca.pkl')
joblib.dump(preprocessor, 'health_preprocessor.pkl')
print("Models and transformers saved.")
