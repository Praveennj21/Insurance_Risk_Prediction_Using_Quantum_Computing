from flask import Flask, render_template, request, redirect, url_for, send_file,session,flash,jsonify
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from werkzeug.utils import secure_filename
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
import logging
import mysql.connector
from functools import wraps
import hashlib
import traceback
import bcrypt

# Load the trained models and preprocessors for both insurance types
vehicle_model = joblib.load('quantum_kernel_random_forest_classifier.pkl')
vehicle_preprocessor = joblib.load('preprocessor.pkl')
vehicle_svd = joblib.load('svd_transformer.pkl')

health_model = joblib.load('health_model.pkl')
health_preprocessor = joblib.load('health_preprocessor.pkl')
health_svd = joblib.load('health_svd.pkl')
n_components = joblib.load('health_n_components.pkl')
health_pca = joblib.load('health_pca.pkl')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)





db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',  # Replace with your MySQL root password
    'database': 'insurance_app'
}


# IBM Quantum Initialization
# Load sensitive information from environment variables
 # Load token from an environment variable

# Global flag to check IBM Quantum server accessibility
ibm_server_accessible = False

try:
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token="34966bcb9ca3e64b91003afca8d9dab299499a7d079fc6f6c2ce9524b17921cba09da884d99a45a8e5f55167dc5e14694bb4c47a2237e9a3c9da5892c87fd4a2"  # Use the actual token loaded from the environment
    )
    # Check available backends
    available_backends = service.backends()
    print("Available backends:", available_backends)
    ibm_server_accessible = True
except Exception as e:
    print("IBM Quantum service could not be initialized:", e)
    ibm_server_accessible = False







# Utility function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_vehicle_amount(data):
    base_amount = 500  # Base insurance amount

    # Risk-based adjustments
    if data['accident_history'] > 2 or data['traffic_violations'] > 2 or data['claims_history'] > 2:
        risk_level = 'High Risk'
    elif data['accident_history'] > 0 or data['traffic_violations'] > 0 or data['claims_history'] > 0:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'Low Risk'

    risk_adjustments = {
        'Low Risk': 300,
        'Moderate Risk': 200,
        'High Risk': 100
    }

    # Vehicle make categorization
    high_level_vehicles = ['BMW', 'Mercedes', 'Audi', 'Porsche']
    medium_level_vehicles = ['Toyota', 'Honda', 'Ford', 'Chevrolet']
    low_level_vehicles = ['Hyundai', 'Kia', 'Nissan', 'Suzuki']

    if data['vehicle_make'] in high_level_vehicles:
        vehicle_make_adjustment = 150  # High-level vehicles charge
    elif data['vehicle_make'] in medium_level_vehicles:
        vehicle_make_adjustment = 75   # Medium-level vehicles charge
    elif data['vehicle_make'] in low_level_vehicles:
        vehicle_make_adjustment = 25   # Low-level vehicles charge
    else:
        vehicle_make_adjustment = 0    # Default, no charge for unlisted vehicles

    # Specific adjustments based on individual factors in the data
    adjustments = {
        'age': 200 if data['age'] < 25 else -50,  # Charge if age is under 25, discount otherwise
        'annual_income': 100 if data['annual_income'] < 30000 else -50,  # Charge for low income
        'vehicle_age': 150 if data['vehicle_age'] > 10 else 0,  # Charge for older vehicles
        'engine_size': 200 if data['engine_size'] > 3.0 else 0,  # Charge for larger engines
        'mileage_driven_annually': 100 if data['mileage_driven_annually'] > 15000 else 0,  # Charge for high mileage
        'accident_history': -data['accident_history'] * 100,  # Discount for each accident (more accidents reduce amount)
        'traffic_violations': data['traffic_violations'] * 50,  # Charge for each traffic violation
        'claims_history': data['claims_history'] * 75,  # Charge for each past claim
        'license_duration': -100 if data['license_duration'] > 10 else 50,  # Discount for experienced drivers
        'residential_location': 100 if data['residential_location'] == 'Urban' else -50,  # Charge for urban, discount for rural
        'vehicle_make': vehicle_make_adjustment
    }

    # Debugging: Print the accident history adjustment
    print(f"Accident History: {data['accident_history']}, Adjustment: {adjustments['accident_history']}")

    # Calculate the total insurance amount
    total_amount = base_amount + risk_adjustments[risk_level] + sum(adjustments.values())
    print(f"Total Amount: {total_amount}")
    return total_amount



def calculate_health_amount(data, risk_level):
        # Base amount and risk level adjustments
        base_amount = 800
        risk_adjustments = {'Low Risk': 300, 'Moderate Risk': 200, 'High Risk': 100}

        # Initialize total adjustments
        total_adjustments = 0

        # Age adjustment
        if data.get('age', 0) > 60:
            total_adjustments += 500
        elif 40 <= data.get('age', 0) <= 60:
            total_adjustments += 300
        else:
            total_adjustments += 100  # Younger individuals

        # BMI adjustment
        if data.get('bmi', 0) > 30:
            total_adjustments += 200
        elif 25 <= data.get('bmi', 0) <= 30:
            total_adjustments += 100

        # Cholesterol adjustment
        if data.get('cholesterol', 0) > 240:
            total_adjustments += 300
        elif 200 <= data.get('cholesterol', 0) <= 240:
            total_adjustments += 150

        # Blood pressure adjustment
        if data.get('blood_pressure', 0) > 140:
            total_adjustments += 400
        elif 120 <= data.get('blood_pressure', 0) <= 140:
            total_adjustments += 200

        # Annual income adjustment
        if data.get('annual_income', 0) > 100000:
            total_adjustments -= 200
        elif 50000 <= data.get('annual_income', 0) <= 100000:
            total_adjustments -= 100

        # Annual claims adjustment
        if data.get('annual_claims', 0) > 3:
            total_adjustments += 500
        elif 1 <= data.get('annual_claims', 0) <= 3:
            total_adjustments += 200

        # Doctor visits adjustment
        if data.get('num_doctor_visits', 0) > 5:
            total_adjustments += 300

        # Specialist visits adjustment
        if data.get('num_specialist_visits', 0) > 3:
            total_adjustments += 400

        # Gender adjustment
        if data.get('gender', '').lower() == 'male':
            total_adjustments += 100
        elif data.get('gender', '').lower() == 'female':
            total_adjustments += 50

        # Smoking status adjustment
        if data.get('smoking_status', '').lower() == 'current smoker':
            total_adjustments += 500
        elif data.get('smoking_status', '').lower() == 'former smoker':
            total_adjustments += 200

        # Exercise frequency adjustment
        if data.get('exercise_frequency', '') == 'Regularly':
            total_adjustments -= 200
        elif data.get('exercise_frequency', '') == 'Occasionally':
            total_adjustments += 100

        # Alcohol consumption adjustment
        if data.get('alcohol_consumption', '') == 'Frequent':
            total_adjustments += 300
        elif data.get('alcohol_consumption', '') == 'Occasional':
            total_adjustments += 150

        # Pre-existing conditions adjustment
        pre_existing_conditions = data.get('pre_existing_conditions', 0)
        if isinstance(pre_existing_conditions, (int, float)) and pre_existing_conditions > 0:
            total_adjustments += pre_existing_conditions * 50

        # Marital status adjustment
        if data.get('marital_status', '').lower() == 'married':
            total_adjustments -= 100
        elif data.get('marital_status', '').lower() == 'single':
            total_adjustments += 50

        # Residential area adjustment
        if data.get('residential_area', '').lower() == 'urban':
            total_adjustments += 200
        elif data.get('residential_area', '').lower() == 'rural':
            total_adjustments -= 50

        # Family medical history adjustment
        family_medical_history = data.get('family_medical_history', '')
        if family_medical_history:  # If non-empty, assume medical history exists
            total_adjustments += 300

        # Healthcare access adjustment
        if data.get('healthcare_access', '').lower() == 'excellent':
            total_adjustments -= 200
        elif data.get('healthcare_access', '').lower() == 'poor':
            total_adjustments += 100

        # Risk adjustment
        risk_adjustment = risk_adjustments.get(risk_level, 0)

        # Calculate total amount
        total_amount = base_amount + risk_adjustment + total_adjustments
        return total_amount

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Hash the password securely
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Check if username already exists
        connection = mysql.connector.connect(**db_config)  # Use the config directly
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        if user:
            flash('Username already exists. Please choose another one.', 'danger')
        else:
            # Insert new user into the database, created_at will be set automatically
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', (username, password_hash))
            connection.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

        cursor.close()
        connection.close()

    return render_template('register.html')

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    # Clear session data when accessing the login page
    session.clear()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Authenticate the user
        connection = mysql.connector.connect(**db_config)  # Use the config directly
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        if user and user[2] == password_hash:  # user[2] is the password_hash column
            session['logged_in'] = True
            session['username'] = username  # Store username in session for personalization
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

        cursor.close()
        connection.close()

    return render_template('login.html')


# Decorator to require login for certain routes
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session and session['logged_in']:
            return f(*args, **kwargs)
        else:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
    return wrap

# Route for the index page
@app.route('/index')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

# Route for logout
@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Prevent browser caching to enforce login requirement on back button
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/change_account', methods=['GET', 'POST'])
@login_required
def change_account():
    if request.method == 'POST':
        # Get form inputs
        current_password = request.form.get('current_password')
        new_username = request.form.get('new_username')
        new_password = request.form.get('new_password')
        username = session.get('username')  # Current logged-in username

        # Hash the current password for verification
        current_password_hash = hashlib.sha256(current_password.encode()).hexdigest()

        try:
            # Connect to the database
            with mysql.connector.connect(**db_config) as connection:
                with connection.cursor() as cursor:
                    # Verify the current password
                    cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
                    user = cursor.fetchone()

                    if not user or user[2] != current_password_hash:  # user[2] is password_hash
                        flash('Current password is incorrect.', 'danger')
                        return render_template('change_account.html')

                    # Update username if provided
                    if new_username:
                        cursor.execute('SELECT * FROM users WHERE username = %s', (new_username,))
                        if cursor.fetchone():
                            flash('Username already exists. Please choose another.', 'danger')
                        else:
                            cursor.execute('UPDATE users SET username = %s WHERE username = %s', (new_username, username))
                            session['username'] = new_username  # Update session
                            flash('Username updated successfully!', 'success')

                    # Update password if provided
                    if new_password:
                        new_password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                        cursor.execute('UPDATE users SET password_hash = %s WHERE username = %s', (new_password_hash, username))
                        flash('Password updated successfully!', 'success')

                # Commit the changes to the database
                connection.commit()

        except mysql.connector.Error as err:
            flash(f'Database error: {err}', 'danger')

    # Render the change_account page
    return render_template('change_account.html')




@app.route('/vehicle_dashboard')
def vehicle_dashboard():
    return render_template('vehicle_dashboard.html')

@app.route('/health_dashboard')
def health_dashboard():
    return render_template('health_dashboard.html')


@app.route('/vehicle_data', methods=['GET', 'POST'])
def vehicle_data():
    if request.method == 'POST':
        try:
            # Parse input data
            if request.is_json:
                data = request.get_json()
            else:
                data = {
                    'age': float(request.form.get('age', 0)),
                    'annual_income': float(request.form.get('annual_income', 0)),
                    'vehicle_age': float(request.form.get('vehicle_age', 0)),
                    'engine_size': float(request.form.get('engine_size', 0)),
                    'mileage_driven_annually': float(request.form.get('mileage_driven_annually', 0)),
                    'accident_history': float(request.form.get('accident_history', 0)),
                    'traffic_violations': float(request.form.get('traffic_violations', 0)),
                    'claims_history': float(request.form.get('claims_history', 0)),
                    'license_duration': float(request.form.get('license_duration', 0)),
                    'gender': request.form.get('gender', ''),
                    'residential_location': request.form.get('residential_location', ''),
                    'vehicle_make': request.form.get('vehicle_make', '')
                }

            # Backend Validation
            if not (18 <= data['age'] <= 75):
                return {"error": "Age must be between 18 and 75."}, 400
            if data['annual_income'] < 20000:
                return {"error": "Annual income must be at least 20,000."}, 400
            if len(data['vehicle_make']) > 20:
                return {"error": "Vehicle make cannot exceed 20 characters."}, 400
            if not (0 <= data['vehicle_age'] <= 20):
                return {"error": "Vehicle age must be between 0 and 20 years."}, 400
            if not (0.1 <= data['engine_size'] <= 5.0):
                return {"error": "Engine size must be between 0.1 and 5.0 liters."}, 400
            if not (0 <= data['mileage_driven_annually'] <= 50000):
                return {"error": "Mileage driven annually must be between 0 and 50,000 km."}, 400
            for field in ['accident_history', 'traffic_violations', 'claims_history']:
                if not (0 <= data[field] <= 10):
                    return {"error": f"{field.replace('_', ' ').capitalize()} must be between 0 and 10."}, 400
            if not (1 <= data['license_duration'] <= 60):
                return {"error": "License duration must be between 1 and 60 years."}, 400
            if data['gender'] not in ['Male', 'Female', 'Other']:
                return {"error": "Gender must be Male, Female, or Other."}, 400
            if data['residential_location'] not in ['Urban', 'Suburban', 'Rural']:
                return {"error": "Residential location must be Urban, Suburban, or Rural."}, 400

            # Preprocess data
            df = pd.DataFrame([data])
            preprocessed = vehicle_preprocessor.transform(df)
            kernel_vector = vehicle_svd.transform(preprocessed)
            risk_level = vehicle_model.predict(kernel_vector)[0]
            insurance_amount = calculate_vehicle_amount(data)

            # Save to database
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO vehicle_data (
                    age, annual_income, vehicle_age, engine_size,
                    mileage_driven_annually, accident_history,
                    traffic_violations, claims_history, license_duration,
                    gender, residential_location, vehicle_make,
                    risk_level, insurance_amount
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                data['age'], data['annual_income'], data['vehicle_age'], data['engine_size'],
                data['mileage_driven_annually'], data['accident_history'],
                data['traffic_violations'], data['claims_history'], data['license_duration'],
                data['gender'], data['residential_location'], data['vehicle_make'],
                risk_level, insurance_amount
            ))
            connection.commit()
            cursor.close()
            connection.close()

            # Return response based on the request type
            if request.is_json:
                return {
                    "risk_level": risk_level,
                    "insurance_amount": insurance_amount
                }

            # Redirect for form-based submissions
            return redirect(url_for('vehicle_result', risk_level=risk_level, insurance_amount=insurance_amount))

        except Exception as e:
            # Log the error and return a user-friendly message
            print(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}, 500

    # Render the form page for GET requests
    return render_template('vehicle_data.html')

@app.route('/vehicle_result', methods=['GET', 'POST'])
def vehicle_result():
    risk_level = request.args.get('risk_level')
    insurance_amount = request.args.get('insurance_amount')
    return render_template('vehicle_result.html', risk_level=risk_level, insurance_amount=insurance_amount)

@app.route('/vehicle_upload', methods=['GET', 'POST'])
def upload_vehicle_file():
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            return render_template('vehicle_upload.html', error_message="No file part in the request.")
        
        file = request.files['file']
        
        # Check if a file is selected
        if file.filename == '':
            return render_template('vehicle_upload.html', error_message="No file selected.")
        
        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return render_template('vehicle_upload.html', error_message="Invalid file type. Please upload a CSV file.")
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Define the required columns
        required_columns = [
            'age', 'annual_income', 'vehicle_age', 'engine_size',
            'mileage_driven_annually', 'accident_history',
            'traffic_violations', 'claims_history', 'license_duration',
            'gender', 'residential_location', 'vehicle_make'
        ]

        try:
            # Load the CSV file
            input_data = pd.read_csv(file_path)
        except Exception as e:
            return render_template('vehicle_upload.html', error_message=f"Error reading CSV file: {str(e)}")

        # Validate the columns
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            return render_template('vehicle_upload.html', error_message=f"Invalid CSV file. Missing columns: {', '.join(missing_columns)}")
        
        try:
            # Backend validation for each row
            for index, row in input_data.iterrows():
                if not (18 <= row['age'] <= 75):
                    return render_template('vehicle_upload.html', error_message=f"Age must be between 18 and 75 at row {index + 1}.")
                if row['annual_income'] < 20000:
                    return render_template('vehicle_upload.html', error_message=f"Annual income must be at least 20,000 at row {index + 1}.")
                if not (0 <= row['vehicle_age'] <= 20):
                    return render_template('vehicle_upload.html', error_message=f"Vehicle age must be between 0 and 20 years at row {index + 1}.")
                if not (0.1 <= row['engine_size'] <= 5.0):
                    return render_template('vehicle_upload.html', error_message=f"Engine size must be between 0.1 and 5.0 liters at row {index + 1}.")
                if not (0 <= row['mileage_driven_annually'] <= 50000):
                    return render_template('vehicle_upload.html', error_message=f"Mileage driven annually must be between 0 and 50,000 km at row {index + 1}.")
                for field in ['accident_history', 'traffic_violations', 'claims_history']:
                    if not (0 <= row[field] <= 10):
                        return render_template('vehicle_upload.html', error_message=f"{field.replace('_', ' ').capitalize()} must be between 0 and 10 at row {index + 1}.")
                if not (1 <= row['license_duration'] <= 60):
                    return render_template('vehicle_upload.html', error_message=f"License duration must be between 1 and 60 years at row {index + 1}.")
                if row['gender'] not in ['M', 'F']:
                    return render_template('vehicle_upload.html', error_message=f"Gender must be Male, Female, or Other at row {index + 1}.")
                if row['residential_location'] not in ['Urban', 'Suburban', 'Rural']:
                    return render_template('vehicle_upload.html', error_message=f"Residential location must be Urban, Suburban, or Rural at row {index + 1}.")
            
            # Preprocess and predict
            preprocessed = vehicle_preprocessor.transform(input_data)
            kernel_vectors = vehicle_svd.transform(preprocessed)
            input_data['risk_level'] = vehicle_model.predict(kernel_vectors)
            
            # Calculate insurance amount
            input_data['insurance_amount'] = input_data.apply(
                lambda row: calculate_vehicle_amount(row), axis=1
            )
            
            # Save the processed file
            output_file = 'vehicle_predictions_output.csv'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
            input_data.to_csv(output_path, index=False)
            
            # Render success template with download link
            return render_template('vehicle_upload.html', download_link=True, filename=output_file)
        
        except Exception as e:
            return render_template('vehicle_upload.html', error_message=f"An error occurred during processing: {str(e)}")

    return render_template('vehicle_upload.html', insurance_type='vehicle')


@app.route('/download/vehicle/<filename>', methods=['GET'])
def download_vehicle_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return render_template('vehicle_upload.html', error_message="The requested file does not exist.")






@app.route('/health_data', methods=['GET', 'POST'])
def health_data():
    if request.method == 'POST':
        try:
            # Parse input data
            if request.is_json:
                data = request.get_json()
            else:
                data = {
                    'age': float(request.form.get('age', 0)),
                    'bmi': float(request.form.get('bmi', 0)),
                    'cholesterol': float(request.form.get('cholesterol', 0)),
                    'blood_pressure': float(request.form.get('blood_pressure', 0)),
                    'annual_income': float(request.form.get('annual_income', 0)),
                    'annual_claims': float(request.form.get('annual_claims', 0)),
                    'num_doctor_visits': float(request.form.get('num_doctor_visits', 0)),
                    'num_specialist_visits': float(request.form.get('num_specialist_visits', 0)),
                    'gender': request.form.get('gender', ''),
                    'smoking_status': request.form.get('smoking_status', ''),
                    'exercise_frequency': request.form.get('exercise_frequency', ''),
                    'alcohol_consumption': request.form.get('alcohol_consumption', ''),
                    'pre_existing_conditions': float(request.form.get('pre_existing_conditions', 0)),
                    'marital_status': request.form.get('marital_status', ''),
                    'residential_area': request.form.get('residential_area', ''),
                    'family_medical_history': request.form.get('family_medical_history', ''),
                    'healthcare_access': request.form.get('healthcare_access', '')
                }

            # Backend validation
            if not (18 <= data['age'] <= 120):
                return {"error": "Age must be between 18 and 120."}, 400
            if not (10 <= data['bmi'] <= 60):
                return {"error": "BMI must be between 10 and 60."}, 400
            if not (100 <= data['cholesterol'] <= 300):
                return {"error": "Cholesterol levels must be between 100 and 300 mg/dL."}, 400
            if not (60 <= data['blood_pressure'] <= 200):
                return {"error": "Blood pressure must be between 60 and 200 mmHg."}, 400
            if not (1000 <= data['annual_income'] <= 1000000):
                return {"error": "Annual income must be between 1,000 and 1,000,000 USD."}, 400
            if not (0 <= data['annual_claims'] <= 1000):
                return {"error": "Annual claims must be between 0 and 1000."}, 400
            if not (0 <= data['num_doctor_visits'] <= 50):
                return {"error": "Number of doctor visits must be between 0 and 50."}, 400
            if not (0 <= data['num_specialist_visits'] <= 50):
                return {"error": "Number of specialist visits must be between 0 and 50."}, 400
            if data['gender'] not in ['Male', 'Female', 'Other']:
                return {"error": "Gender must be 'Male', 'Female', or 'Other'."}, 400
            if data['smoking_status'] not in ['Non-Smoker', 'Former Smoker', 'Current Smoker']:
                return {"error": "Smoking status must be 'Non-Smoker', 'Former Smoker', or 'Current Smoker'."}, 400
            if data['exercise_frequency'] not in ['None', 'Occasionally', 'Regularly']:
                return {"error": "Exercise frequency must be 'None', 'Occasionally', or 'Regularly'."}, 400
            if data['alcohol_consumption'] not in ['None', 'Occasional', 'Frequent']:
                return {"error": "Alcohol consumption must be 'None', 'Occasional', or 'Frequent'."}, 400
            if not (0 <= data['pre_existing_conditions'] <= 10):
                return {"error": "Pre-existing conditions must be between 0 and 10."}, 400
            if data['marital_status'] not in ['Single', 'Married', 'Divorced']:
                return {"error": "Marital status must be 'Single', 'Married', or 'Divorced'."}, 400
            if data['residential_area'] not in ['Urban', 'Suburban', 'Rural']:
                return {"error": "Residential area must be 'Urban', 'Suburban', or 'Rural'."}, 400
            if data['healthcare_access'] not in ['Good', 'Average', 'Poor']:
                return {"error": "Healthcare access must be 'Good', 'Average', or 'Poor'."}, 400

            # Preprocess data
            input_df = pd.DataFrame([data])
            preprocessed_input = health_preprocessor.transform(input_df)
            svd_input = health_svd.transform(preprocessed_input)
            pca_input = health_pca.transform(svd_input)

            # Make prediction
            risk_level = health_model.predict(pca_input)[0]
            insurance_amount = calculate_health_amount(data, risk_level)

            # Save to database
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO health_data (
                    age, bmi, cholesterol, blood_pressure, annual_income,
                    annual_claims, num_doctor_visits, num_specialist_visits,
                    gender, smoking_status, exercise_frequency, alcohol_consumption,
                    pre_existing_conditions, marital_status, residential_area,
                    family_medical_history, healthcare_access, risk_level, insurance_amount
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                data['age'], data['bmi'], data['cholesterol'], data['blood_pressure'], data['annual_income'],
                data['annual_claims'], data['num_doctor_visits'], data['num_specialist_visits'],
                data['gender'], data['smoking_status'], data['exercise_frequency'], data['alcohol_consumption'],
                data['pre_existing_conditions'], data['marital_status'], data['residential_area'],
                data['family_medical_history'], data['healthcare_access'], risk_level, insurance_amount
            ))
            connection.commit()
            cursor.close()
            connection.close()

            # Return response based on request type
            if request.is_json:
                return {
                    "risk_level": risk_level,
                    "insurance_amount": insurance_amount
                }

            return redirect(url_for('helath_result', risk_level=risk_level, insurance_amount=insurance_amount))

        except Exception as e:
            print(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}, 500

    return render_template('health_data.html')




@app.route('/health_result', methods=['GET', 'POST'])
def helath_result():
    risk_level = request.args.get('risk_level')
    insurance_amount = request.args.get('insurance_amount')
    return render_template('helath_result.html', risk_level=risk_level, insurance_amount=insurance_amount)





@app.route('/health_upload', methods=['GET', 'POST'])
def upload_health_file():
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            return render_template('health_upload.html', error_message="No file part in the request.")
        
        file = request.files['file']

        # Check if a file is selected
        if file.filename == '':
            return render_template('health_upload.html', error_message="No file selected.")
        
        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return render_template('health_upload.html', error_message="Invalid file type. Please upload a CSV file.")
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Define the required columns
        required_columns = [
            'age', 'bmi', 'cholesterol', 'blood_pressure', 'annual_income',
            'annual_claims', 'num_doctor_visits', 'num_specialist_visits',
            'gender', 'smoking_status', 'exercise_frequency',
            'alcohol_consumption', 'marital_status', 'residential_area',
            'healthcare_access'
        ]

        try:
            # Load the CSV file
            input_data = pd.read_csv(file_path)
        except Exception as e:
            return render_template('health_upload.html', error_message=f"Error reading CSV file: {str(e)}")

        # Validate the columns
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            return render_template('health_upload.html', error_message=f"Invalid CSV file. Missing columns: {', '.join(missing_columns)}")

        try:
            # Backend validation for each row
            for index, row in input_data.iterrows():
                if not (18 <= row['age'] <= 120):
                    return render_template('health_upload.html', error_message=f"Age must be between 18 and 120 at row {index + 1}.")
                if not (10 <= row['bmi'] <= 60):
                    return render_template('health_upload.html', error_message=f"BMI must be between 10 and 60 at row {index + 1}.")
                if not (100 <= row['cholesterol'] <= 300):
                    return render_template('health_upload.html', error_message=f"Cholesterol must be between 100 and 300 mg/dL at row {index + 1}.")
                if not (60 <= row['blood_pressure'] <= 200):
                    return render_template('health_upload.html', error_message=f"Blood pressure must be between 60 and 200 mmHg at row {index + 1}.")
                if not (1000 <= row['annual_income'] <= 1000000):
                    return render_template('health_upload.html', error_message=f"Annual income must be between 1,000 and 1,000,000 USD at row {index + 1}.")
                for field, max_value in [('annual_claims', 10000), ('num_doctor_visits', 50), ('num_specialist_visits', 50)]:
                    if not (0 <= row[field] <= max_value):
                        return render_template('health_upload.html', error_message=f"{field.replace('_', ' ').capitalize()} must be between 0 and {max_value} at row {index + 1}.")
                for field, valid_values in [
                    ('gender', ['M', 'F']),
                    ('smoking_status', ['Non-smoker', 'Smoker']),
                    ('exercise_frequency', ['Never', 'Occasionally', 'Regularly']),
                    ('alcohol_consumption', ['Never', 'Occasionally', 'Frequently']),
                    ('marital_status', ['Single', 'Married', 'Divorced', 'Widowed']),
                    ('residential_area', ['Urban', 'Suburban', 'Rural']),
                    ('healthcare_access', ['Good', 'Average', 'Poor', 'Excellent'])
                ]:
                    if row[field] not in valid_values:
                        return render_template('health_upload.html', error_message=f"{field.replace('_', ' ').capitalize()} must be one of {', '.join(valid_values)} at row {index + 1}.")

            # Preprocess and predict
            preprocessed = health_preprocessor.transform(input_data)
            kernel_vectors = health_svd.transform(preprocessed)
            pca_vectors = health_pca.transform(kernel_vectors)
            input_data['risk_level'] = health_model.predict(pca_vectors)

            # Calculate insurance amount
            input_data['insurance_amount'] = input_data.apply(
                lambda row: calculate_health_amount(row, row['risk_level']), axis=1
            )

            # Save the processed file
            output_file = 'health_predictions_output.csv'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
            input_data.to_csv(output_path, index=False)

            # Render success template with download link
            return render_template('health_upload.html', download_link=True, filename=output_file)

        except Exception as e:
            return render_template('health_upload.html', error_message=f"An error occurred during processing: {str(e)}")

    return render_template('health_upload.html', insurance_type='health')


@app.route('/download/health/<filename>', methods=['GET'])
def download_health_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return render_template('health_upload.html', error_message="The requested file does not exist.")






if __name__ == '__main__':
    app.run(debug=True)