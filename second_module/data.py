import pandas as pd
import numpy as np
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Feature ranges and categories
ages = list(range(18, 70))
genders = ['Male', 'Female']
annual_incomes = list(range(20000, 150001, 5000))
locations = ['Urban', 'Rural', 'Suburban']
vehicle_makes = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe']
vehicle_ages = list(range(1, 15))
engine_sizes = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
mileages = list(range(5000, 20001, 1000))
accidents = [0, 1, 2, 3,4, 5, 6, 7, 8, 9]
violations = [0, 1, 2, 3,4, 5, 6, 7, 8, 9]
claims = [0, 1, 2, 3,4, 5, 6, 7, 8, 9]
license_durations = list(range(1, 40))

# Target samples for balanced distribution
target_samples_per_level = 100

# Helper function to generate samples with meaningful correlations
def generate_sample():
    age = random.choice(ages)
    gender = random.choice(genders)
    annual_income = random.choice(annual_incomes)
    residential_location = random.choice(locations)
    vehicle_make = random.choice(vehicle_makes)
    vehicle_age = random.choice(vehicle_ages)
    engine_size = random.choice(engine_sizes)
    mileage_driven_annually = random.choice(mileages)
    accident_history = random.choice([0, 1, 2, 3] if age < 30 else [2, 3])
    traffic_violations = random.choice([0, 1, 2] if residential_location == 'Rural' else [2, 3, 4])
    claims_history = random.choice(claims)
    license_duration = random.choice(license_durations)

    # Calculate risk level based on correlations
    if (annual_income < 40000 and accident_history >= 3 and traffic_violations > 4) or \
       (vehicle_age > 10 and mileage_driven_annually > 15000) or \
       (claims_history >= 3 and residential_location == 'Urban'):
        risk_level = 'High'
    elif (annual_income < 70000 or accident_history >= 2 or traffic_violations > 2) and \
         (vehicle_age > 7 or engine_size >= 2.5):
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {
        'age': age,
        'gender': gender,
        'annual_income': annual_income,
        'residential_location': residential_location,
        'vehicle_make': vehicle_make,
        'vehicle_age': vehicle_age,
        'engine_size': engine_size,
        'mileage_driven_annually': mileage_driven_annually,
        'accident_history': accident_history,
        'traffic_violations': traffic_violations,
        'claims_history': claims_history,
        'license_duration': license_duration,
        'risk_level': risk_level,
    }

# Generate a balanced dataset for each risk level
def generate_data(target_samples):
    all_samples = []
    risk_levels = ['High', 'Medium', 'Low']
    risk_count = {level: 0 for level in risk_levels}

    while any(risk_count[level] < target_samples for level in risk_levels):
        sample = generate_sample()
        if risk_count[sample['risk_level']] < target_samples:
            all_samples.append(sample)
            risk_count[sample['risk_level']] += 1
    
    return all_samples

# Generate and save the data
all_samples = generate_data(target_samples_per_level)
df = pd.DataFrame(all_samples)

# Save to CSV
csv_file_path = 'precise_insurance_data2.csv'
df.to_csv(csv_file_path, index=False)

print(f"Generated data successfully saved to {csv_file_path}")

# Verify the saved data by loading it and printing a preview
saved_df = pd.read_csv(csv_file_path)
print("Data Preview after saving:")
print(saved_df.head())
print(f"Number of rows: {len(saved_df)}")
print("Risk level distribution:\n", saved_df['risk_level'].value_counts())
