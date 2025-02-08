import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

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

# Load your training data
training_data = pd.read_csv('precise_insurance_data2.csv')

# Preprocess the training data
preprocessed_data = preprocessor.fit_transform(training_data)

# Save the preprocessor
joblib.dump(preprocessor, 'vehicle_preprocessor.pkl')

# Fit and save SVD
n_components = min(240, preprocessed_data.shape[1])  # Adjust based on feature count
svd = TruncatedSVD(n_components=n_components)
svd.fit(preprocessed_data)

# Save the SVD transformer
joblib.dump(svd, 'vehicle_svd.pkl')
print("Preprocessor and SVD transformer saved.")
