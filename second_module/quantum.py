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
from qiskit.quantum_info import Statevector
from joblib import dump, Parallel, delayed

# Step 1: Load the Data
data = pd.read_csv('precise_insurance_data2.csv')

# Data Preview
print("Data Preview:")
print(data.head())

# Step 2: Data Preprocessing
features = data.drop(['risk_level'], axis=1)  # Features
target = data['risk_level'].map({'High': 2, 'Medium': 1, 'Low': 0})  # Map classes to integers

numeric_features = [col for col in features.columns if features[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in features.columns if features[col].dtype == 'object']

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Step 5: Preprocess the Data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 6: Define the Quantum Kernel Function
def variational_quantum_kernel(x1, x2):
    n_qubits = len(x1)
    circuit = QuantumCircuit(n_qubits)
    params = [Parameter(f'θ{i}') for i in range(n_qubits)]

    # Apply multiple rotation layers and entanglement
    for _ in range(2):
        for i in range(n_qubits):
            circuit.ry(params[i], i)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)

    # Assign parameters and evolve statevectors
    circuit_x1 = circuit.assign_parameters({params[i]: x1[i] for i in range(n_qubits)}, inplace=False)
    circuit_x2 = circuit.assign_parameters({params[i]: x2[i] for i in range(n_qubits)}, inplace=False)

    state1 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x1)
    state2 = Statevector.from_int(0, 2**n_qubits).evolve(circuit_x2)

    # Return squared magnitude of dot product
    return np.abs(state1.data.conj().dot(state2.data))**2

# Step 7: Compute Quantum Kernel Matrix (With Respect to Training Data)
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

# Compute Kernel Matrices (Training and Testing)
print("Computing the quantum kernel matrix for training data...")
X_train_vqkernel = compute_variational_kernel(X_train_preprocessed, X_train_preprocessed)

print("Computing the quantum kernel matrix for test data...")
X_test_vqkernel = compute_variational_kernel(X_test_preprocessed, X_train_preprocessed)

# Step 8: Use SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vqkernel, y_train)

# Verify shapes after SMOTE
print(f'Shape after SMOTE - Features: {X_train_resampled.shape}, Target: {y_train_resampled.shape}')

# Step 9: Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)

classifier = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')

# Step 10: Make Predictions and Evaluate
y_pred = classifier.predict(X_test_vqkernel)

# Step 11: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with Variational Quantum Kernel: {accuracy:.2f}')

print("Classification Report:\n", 
      classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], zero_division=1))

# Step 12: Save the Trained Model
dump(classifier, 'vehicle_insurance_model.pkl')
print("Model saved as 'vehicle_insurance_model.pkl'")
