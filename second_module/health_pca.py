import pandas as pd
from sklearn.decomposition import PCA
import joblib
import pickle

def train_and_save_pca_model(X, n_components):
    """Train and save a PCA model with the specified number of components."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Save the PCA model to disk
    joblib.dump(pca, 'health_pca_model.pkl')

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each component: {explained_variance}")

    print("PCA model saved successfully.")

if __name__ == "__main__":
    # Paths to your data and preprocessing model
    data_path = r'pattern_based_health_insurance_risk_data.csv'
    preprocessing_model_path = r'health_preprocessor.pkl'

    # Load the preprocessing model
    with open(preprocessing_model_path, 'rb') as preprocess_file:
        preprocessor = pickle.load(preprocess_file)
    
    # Check if preprocessor is valid
    if not hasattr(preprocessor, 'transform'):
        print("Loaded object is not a valid preprocessor with a 'transform' method.")
        exit(1)

    # Load and preprocess the data
    data = pd.read_csv(data_path)

    # Updated relevant columns
    relevant_columns = [
        'name', 'age', 'gender', 'bmi', 'cholesterol', 'blood_pressure',
        'smoking_status', 'exercise_frequency', 'alcohol_consumption',
        'pre_existing_conditions', 'marital_status', 'annual_income',
        'residential_area', 'family_medical_history', 'healthcare_access',
        'annual_claims', 'num_doctor_visits', 'num_specialist_visits', 'risk_level'
    ]

    # Ensure columns are correctly aligned
    data = data[relevant_columns]

    # Convert categorical variables if necessary
    X = data.drop('risk_level', axis=1)
    X = pd.get_dummies(X)  # Convert categorical variables to numeric (if needed)

    # Apply preprocessing
    try:
        X_preprocessed = preprocessor.transform(X)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        print("Ensure the preprocessing model matches the input features.")
        exit(1)

    # Train PCA only if required
    n_components = min(9, X_preprocessed.shape[1])
    train_and_save_pca_model(X_preprocessed, n_components)
