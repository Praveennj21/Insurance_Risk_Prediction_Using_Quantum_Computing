import requests

# URL of the endpoint you want to test
url = 'http://127.0.0.1:5000/vehicle_result'  # Replace with your actual endpoint

# Adding query parameters to the URL (if needed)
params = {
    'risk_level': 'High',      # Example risk level
    'insurance_amount': 1350   # Example insurance amount
}

# Send GET request
response = requests.get(url, params=params)

# Check the status code to ensure the request was successful
if response.status_code == 200:
    print(f"Request was successful! Response: {response.text}")  # Display the content of the response
else:
    print(f"Request failed with status code {response.status_code}")
