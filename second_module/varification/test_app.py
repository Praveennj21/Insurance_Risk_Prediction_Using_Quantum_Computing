import unittest
from app import app  # Import your Flask app
class FlaskAppTests(unittest.TestCase):

    # Set up the test client
    def setUp(self):
        self.app = app.test_client()  # Create a test client
        self.app.testing = True  # Enable testing mode

    # Test the login page (GET request)
    def test_login(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)  # Check if the status code is 200 (OK)
        self.assertIn(b'Login', response.data)  # Check if the word "Login" is present in the response

    # Test the vehicle data submission (POST request)
    def test_vehicle_data(self):
        data = {
            'age': 30,
            'annual_income': 50000,
            'vehicle_age': 5,
            'engine_size': 1.8,
            'mileage_driven_annually': 15000,
            'accident_history': 0,
            'traffic_violations': 0,
            'claims_history': 0,
            'license_duration': 5,
            'gender': 'Male',
            'residential_location': 'Urban',
            'vehicle_make': 'Toyota'
        }
        response = self.app.post('/vehicle_data', data=data)
        self.assertEqual(response.status_code, 302)  # Check if redirected
        self.assertIn(b'Insurance Amount', response.data)  # Check if insurance amount is in the response

    # Test health data submission (POST request)
    def test_health_data(self):
        data = {
            'age': 45,
            'bmi': 28,
            'cholesterol': 190,
            'blood_pressure': 120,
            'annual_income': 60000,
            'annual_claims': 2,
            'num_doctor_visits': 3,
            'num_specialist_visits': 1,
            'gender': 'Female',
            'smoking_status': 'Non-Smoker',
            'exercise_frequency': 'Regular',
            'alcohol_consumption': 'Moderate',
            'pre_existing_conditions': 0,
            'marital_status': 'Married',
            'residential_area': 'Urban',
            'family_medical_history': 'None',
            'healthcare_access': 'Yes'
        }
        response = self.app.post('/health_data', data=data)
        self.assertEqual(response.status_code, 200)  # Ensure the page renders correctly
        self.assertIn(b'Insurance Amount', response.data)  # Check if insurance amount is shown

if __name__ == '__main__':
    unittest.main()  # Run the tests
