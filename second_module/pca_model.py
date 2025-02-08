import pandas as pd
from sklearn.decomposition import PCA
import joblib
import pickle  # Use pickle to load the preprocessing model

def train_and_save_pca_model(X, n_components):
    """Train and save a PCA model with the specified number of components."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Save the PCA model to disk
    joblib.dump(pca, 'pca_model.pkl')

    # Log the explained variance ratio for each component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each component: {explained_variance}")

    print("PCA model saved successfully.")

if __name__ == "__main__":
    data_path = r'precise_insurance_data2.csv'  # Training data path

    # Step 1: Load the preprocessing model
    with open(r'preprocessing_model.pkl', 'rb') as preprocess_file:
        preprocessor = pickle.load(preprocess_file)

    # Step 2: Load and preprocess the data
    data = pd.read_csv(data_path)

    relevant_columns = [
        'age', 'annual_income', 'vehicle_age', 'engine_size',
        'mileage_driven_annually', 'accident_history', 'traffic_violations',
        'claims_history', 'license_duration', 'risk_level', 'gender',
        'residential_location', 'vehicle_make'
    ]

    # Select relevant features and target for PCA
    data = data[relevant_columns]
    X = data.drop('risk_level', axis=1)  # Drop the target column

    # Apply preprocessing transformations
    X_preprocessed = preprocessor.transform(X)

    # Step 3: Train the PCA model
    n_components = min(9, X_preprocessed.shape[1])  # Ensure n_components <= features
    train_and_save_pca_model(X_preprocessed, n_components)
