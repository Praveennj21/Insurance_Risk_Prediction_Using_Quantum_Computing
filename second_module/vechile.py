import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from joblib import dump, Parallel, delayed
from sklearn.decomposition import PCA, TruncatedSVD
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import OneHotEncoder



# Initialize IBM Quantum Runtime Service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="c5427c304b52a180799f02ba2f4a662f4e5b4cd712a8f0d44d822b66fa217fcdf70497b5211eac4468a7ccfa5f1214a07349346b1f15ceeec74a05216ba0954d"  # Replace with your IBM Quantum API token
)

# Check available backends
available_backends = service.backends()
print("Available backends:", available_backends)

# Use a valid backend from the available ones
backend_name = "ibmq_qasm_simulator"  # Change this to a backend name that exists in your available backends list

# Step 1: Load the Data
data = pd.read_csv('second_module/uploads/generated_health_insurance_risk_data.csv')

# Data Preview
print("Data Preview:")
print(data.head())

# Step 2: Data Preprocessing
features = data.drop(['risk_level'], axis=1)
target = data['risk_level'].map({'High': 2, 'Medium': 1, 'Low': 0})

# Identify numeric and categorical columns
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Preprocess Data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Dimensionality Reduction using PCA
max_components = min(X_train_preprocessed.shape[0], X_train_preprocessed.shape[1])
pca = PCA(n_components=min(max_components, 10))  # Use 10 as fallback or adjust as needed

X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum()}")

# Apply Truncated SVD for further dimensionality reduction
n_components_svd = min(240, X_train_pca.shape[1])  # Adjust to feature count
svd = TruncatedSVD(n_components=n_components_svd)
X_train_svd = svd.fit_transform(X_train_pca)
X_test_svd = svd.transform(X_test_pca)

# Define the Variational Quantum Kernel Function
def variational_quantum_kernel(x1, x2):
    n_qubits = len(x1)
    circuit = QuantumCircuit(n_qubits)
    params = [Parameter(f'θ{i}') for i in range(n_qubits)]

    for _ in range(2):
        for i in range(n_qubits):
            circuit.ry(params[i], i)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)

    circuit_x1 = circuit.assign_parameters({params[i]: x1[i] for i in range(n_qubits)}, inplace=False)
    circuit_x2 = circuit.assign_parameters({params[i]: x2[i] for i in range(n_qubits)}, inplace=False)

    state1 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x1)
    state2 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x2)

    return np.abs(state1.data.conj().dot(state2.data))**2

# Compute Quantum Kernel Matrix
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

# Compute Kernel Matrices
print("Computing the quantum kernel matrix for training data...")
X_train_vqkernel = compute_variational_kernel(X_train_svd, X_train_svd)

print("Computing the quantum kernel matrix for test data...")
X_test_vqkernel = compute_variational_kernel(X_test_svd, X_train_svd)

# Use SMOTE to Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vqkernel, y_train)

print(f'Shape after SMOTE - Features: {X_train_resampled.shape}, Target: {y_train_resampled.shape}')

# Train RandomForest on Kernel Data
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best Estimator
classifier = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')

# Predict and Evaluate
y_pred = classifier.predict(X_test_vqkernel)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with Variational Quantum Kernel: {accuracy:.2f}')

print("Classification Report:\n", 
      classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], zero_division=1))