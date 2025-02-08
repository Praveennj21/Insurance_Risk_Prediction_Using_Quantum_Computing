import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA, TruncatedSVD
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from joblib import dump, Parallel, delayed

# Step 1: Initialize IBM Quantum Service
TOKEN = "34aa5595c8bc070c0f896b4d3200b40d753df0dc3cb196080e525d86daaf1af835027e233e0ff48f178e9a401510d6cf85a2b4512f89a54d5f50889ffb42559f"
service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
available_backends = service.backends()
print("Available IBM Quantum Backends:", available_backends)
backend_name = "ibmq_manila"

# Step 2: Load Dataset
data = pd.read_csv('health_predictions_output.csv')
print("Data Preview:")
print(data.head())

# Step 3: Preprocessing
features = data.drop(['risk_level'], axis=1)
target = data['risk_level'].map({'High': 2, 'Medium': 1, 'Low': 0})

numeric_features = [
    'age', 'bmi', 'cholesterol', 'blood_pressure', 'annual_income',
    'annual_claims', 'num_doctor_visits', 'num_specialist_visits'
]
categorical_features = [
    'gender', 'smoking_status', 'exercise_frequency', 'alcohol_consumption',
    'pre_existing_conditions', 'marital_status', 'residential_area',
    'family_medical_history', 'healthcare_access'
]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 4: Dimensionality Reduction
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)

svd = TruncatedSVD(n_components=min(240, X_train_pca.shape[1]))
X_train_svd = svd.fit_transform(X_train_pca)
X_test_svd = svd.transform(X_test_pca)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.2f}")

# Step 5: Variational Quantum Kernel
def variational_quantum_kernel(x1, x2):
    n_qubits = len(x1)
    circuit = QuantumCircuit(n_qubits)
    params = [Parameter(f'θ{i}') for i in range(n_qubits)]

    for _ in range(2):  # Circuit depth = 2
        for i in range(n_qubits):
            circuit.ry(params[i], i)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)

    circuit_x1 = circuit.assign_parameters({params[i]: x1[i] for i in range(n_qubits)}, inplace=False)
    circuit_x2 = circuit.assign_parameters({params[i]: x2[i] for i in range(n_qubits)}, inplace=False)

    state1 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x1)
    state2 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x2)

    return np.abs(state1.data.conj().dot(state2.data))**2

def compute_variational_kernel(X1, X2):
    n_samples1, n_samples2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n_samples1, n_samples2))

    results = Parallel(n_jobs=-1)(
        delayed(variational_quantum_kernel)(X1[i], X2[j])
        for i in range(n_samples1)
        for j in range(n_samples2)
    )

    index = 0
    for i in range(n_samples1):
        for j in range(n_samples2):
            kernel_matrix[i, j] = results[index]
            index += 1

    return kernel_matrix

X_train_vqkernel = compute_variational_kernel(X_train_svd, X_train_svd)
X_test_vqkernel = compute_variational_kernel(X_test_svd, X_train_svd)

# Step 6: Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vqkernel, y_train)
print(f"SMOTE - Resampled Train Shape: {X_train_resampled.shape}, {y_train_resampled.shape}")

# Step 7: Train Model with Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)

best_classifier = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 8: Evaluate Model
y_pred = best_classifier.predict(X_test_vqkernel)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], zero_division=1))

# Step 9: Save Model and Preprocessing Pipelines
dump(best_classifier, 'health_insurance_model_ibm.pkl')
dump(svd, 'halth_svd.pkl')
dump(preprocessor, 'health_preproceor.pkl')
print("Model and transformers saved.")
