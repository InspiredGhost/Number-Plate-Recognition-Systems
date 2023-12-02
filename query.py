import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('./keys/keys.json')  # Replace with your service account key file
firebase_admin.initialize_app(cred)

# Function to query Firebase and return True if successful
def query_firestore(string):
    # Initialize Firestore client
    db = firestore.client()

    response_date = False

    # Replace 'collection_name' with your actual collection name in Firestore
    docs = db.collection('registrations').where('registration', '==', string).get()
    for doc in docs:
        status = doc.to_dict().get('status')
        citizen_ref = doc.to_dict().get('citizen')
        if status and status != 'ACTIVE':
            with app.app_context():
                response_date = True

        if status and status == 'ACTIVE':
            # Query another collection when registration is 'ACTIVE'
            fines = db.collection('fines').where('citizen', '==', citizen_ref).get()
            fines_len = len(list(fines))
            if fines_len > 0:
                response_date = True
                break

            warrants = db.collection('warrants').where('citizen', '==', citizen_ref).get()
            warrants_len = len(list(warrants))
            if warrants_len > 0:
                response_date = True
                break
    with app.app_context():
       return response_date

result = query_firestore("CY22133")
print(result)
