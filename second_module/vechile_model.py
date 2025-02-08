import os
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

# Initialize Qiskit Runtime Service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="34966bcb9ca3e64b91003afca8d9dab299499a7d079fc6f6c2ce9524b17921cba09da884d99a45a8e5f55167dc5e14694bb4c47a2237e9a3c9da5892c87fd4a2"
)

# Select Backend
backend_name = "ibm_sherbrooke"  # Replace with your desired backend
backend = service.backend(backend_name)

# Load Dataset
data = pd.read_csv('second_module/vehicle_Train.csv')
print("Data Preview:")
print(data.head())

# Preprocessing
features = data.drop(['risk_level'], axis=1)
target = data['risk_level'].map({'High': 2, 'Medium': 1, 'Low': 0})

# Identify numeric and categorical columns
numeric_features = [col for col in features.columns if features[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in features.columns if features[col].dtype == 'object']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Preprocessing Pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Preprocess Data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Quantum Kernel Function
def variational_quantum_kernel(x1, x2):
    try:
        # Define Quantum Circuit
        n_qubits = len(x1)
        circuit = QuantumCircuit(n_qubits)
        params = [Parameter(f'θ{i}') for i in range(n_qubits)]

        for _ in range(2):  # Add depth to the circuit
            for i in range(n_qubits):
                circuit.ry(params[i], i)
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)

        # Assign parameters and execute
        with Session(backend=backend) as session:
            sampler = Sampler(session=session)
            job_x1 = sampler.run(circuit.assign_parameters({params[i]: x1[i] for i in range(n_qubits)}))
            job_x2 = sampler.run(circuit.assign_parameters({params[i]: x2[i] for i in range(n_qubits)}))
            state1 = np.array(list(job_x1.result().quasi_dists[0].values()))
            state2 = np.array(list(job_x2.result().quasi_dists[0].values()))

        return np.abs(np.dot(state1, state2)) ** 2
    except Exception as e:
        print(f"Quantum kernel computation error: {e}")
        return 0

# Compute Kernel Matrix
def compute_kernel_matrix(X1, X2):
    n_samples1, n_samples2 = len(X1), len(X2)
    results = Parallel(n_jobs=-1)(
        delayed(variational_quantum_kernel)(X1[i], X2[j])
        for i in range(n_samples1) for j in range(n_samples2)
    )
    return np.array(results).reshape(n_samples1, n_samples2)

# Kernel Matrix Computation
print("Computing quantum kernel matrix for resampled training data...")
X_train_vqkernel = compute_kernel_matrix(X_train_resampled, X_train_resampled)

print("Computing quantum kernel matrix for testing data...")
X_test_vqkernel = compute_kernel_matrix(X_test_preprocessed, X_train_resampled)

# Model Training with Grid Search
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_vqkernel, y_train_resampled)

classifier = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predictions and Evaluation
y_pred = classifier.predict(X_test_vqkernel)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Save Model
dump(classifier, 'vehicle_insurance_model_ibm.pkl')
print("Model saved.")
