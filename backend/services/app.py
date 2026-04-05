
from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
import joblib
import data_process

current_file_dir = os.path.dirname(os.path.realpath(__file__)) #Get active file directory
parent_dir = os.path.dirname(current_file_dir) #Get parent of active file directory
os.chdir(os.path.dirname(parent_dir)) #Set active directory to parent of active file directory, sets directory to backend folder

loaded_model = joblib.load('backend/model/trained_model_v2_illness.joblib')

loaded_model_2 = joblib.load('backend/model/trained_model_v2_treat.joblib')

#Python dictionary to hold key value pairs for the reponse sent to the client currently empy defaults, acts as in memory storage holding the last results
res = {"Illness": "",
       "Treatment": ""}

cols = ['age', 'symptom_symptom_token_a cough that lasts more than three weeks', 'symptom_symptom_token_a dry', 
        'symptom_symptom_token_allergy', 'symptom_symptom_token_bluish skin', 'symptom_symptom_token_breath', 
        'symptom_symptom_token_chest congestion', 'symptom_symptom_token_chest pain', 'symptom_symptom_token_chills', 
        'symptom_symptom_token_chronic cough', 'symptom_symptom_token_cold', 'symptom_symptom_token_cough', 
        'symptom_symptom_token_cough with blood', 'symptom_symptom_token_coughing', 'symptom_symptom_token_coughing up blood', 
        'symptom_symptom_token_coughing up yellow or green mucus daily', 'symptom_symptom_token_crackling sound in the lungs while breathing in', 
        'symptom_symptom_token_daytime sleepiness', 'symptom_symptom_token_diarrhea', 'symptom_symptom_token_difficulties with memory and concentration', 
        'symptom_symptom_token_distressing', 'symptom_symptom_token_dizziness', 'symptom_symptom_token_dry cough', 'symptom_symptom_token_dry mouth', 
        'symptom_symptom_token_edema', 'symptom_symptom_token_fainting', 'symptom_symptom_token_faster heart beating', 'symptom_symptom_token_fatigue', 
        'symptom_symptom_token_feeling run-down or tired', 'symptom_symptom_token_fever', 'symptom_symptom_token_frequently waking', 
        'symptom_symptom_token_greenish cough', 'symptom_symptom_token_headache', 'symptom_symptom_token_heart palpitations', 
        'symptom_symptom_token_high fever', 'symptom_symptom_token_irritability', 'symptom_symptom_token_joint pain', 'symptom_symptom_token_loss of appetite', 
        'symptom_symptom_token_loss of appetite and unintentional weight loss', 'symptom_symptom_token_low energy', 'symptom_symptom_token_low-grade fever', 
        'symptom_symptom_token_lower back pain', 'symptom_symptom_token_morning headaches', 'symptom_symptom_token_mucus', 
        'symptom_symptom_token_muscle aches', 'symptom_symptom_token_nasal congestion', 'symptom_symptom_token_nausea', 'symptom_symptom_token_night sweats', 
        'symptom_symptom_token_pain', 'symptom_symptom_token_pauses in breathing', 'symptom_symptom_token_persistent dry coug', 
        'symptom_symptom_token_persistent dry cough', 'symptom_symptom_token_runny nose', 'symptom_symptom_token_shaking', 'symptom_symptom_token_shallow breathing', 
        'symptom_symptom_token_sharp chest pain', 'symptom_symptom_token_short', 'symptom_symptom_token_short of breath', 'symptom_symptom_token_shortness of breath', 
        'symptom_symptom_token_shortness of breath that gets worse during flare-ups', 'symptom_symptom_token_snoring', 'symptom_symptom_token_sore throat', 
        'symptom_symptom_token_stuffy nose', 'symptom_symptom_token_sweating', 'symptom_symptom_token_tight feeling in the chest', 
        'symptom_symptom_token_unusual moodiness', 'symptom_symptom_token_vomiting', 'symptom_symptom_token_weight loss', 
        'symptom_symptom_token_weight loss from loss of appetite', 'symptom_symptom_token_wheezing', 'symptom_symptom_token_wheezing cough', 
        'symptom_symptom_token_whistling sound while breathing', 'symptom_symptom_token_whistling sound while you breathe', 'symptom_symptom_token_yellow cough', 
        'sex_female', 'sex_male', 'sex_not to say', 'sex_unknown', 'nature_high', 'nature_low', 'nature_medium', 'nature_unknown', 'age_group_child', 
        'age_group_teen', 'age_group_young_adult', 'age_group_adult', 'age_group_senior', 'symptom_count', 'high_risk']

cols_2 = ['age', 'symptom_symptom_token_a cough that lasts more than three weeks', 'symptom_symptom_token_a dry', 'symptom_symptom_token_allergy', 
          'symptom_symptom_token_bluish skin', 'symptom_symptom_token_breath', 'symptom_symptom_token_chest congestion', 'symptom_symptom_token_chest pain', 
          'symptom_symptom_token_chills', 'symptom_symptom_token_chronic cough', 'symptom_symptom_token_cold', 'symptom_symptom_token_cough', 
          'symptom_symptom_token_cough with blood', 'symptom_symptom_token_coughing', 'symptom_symptom_token_coughing up blood', 
          'symptom_symptom_token_coughing up yellow or green mucus daily', 'symptom_symptom_token_crackling sound in the lungs while breathing in', 
          'symptom_symptom_token_daytime sleepiness', 'symptom_symptom_token_diarrhea', 'symptom_symptom_token_difficulties with memory and concentration', 
          'symptom_symptom_token_distressing', 'symptom_symptom_token_dizziness', 'symptom_symptom_token_dry cough', 'symptom_symptom_token_dry mouth', 
          'symptom_symptom_token_edema', 'symptom_symptom_token_fainting', 'symptom_symptom_token_faster heart beating', 'symptom_symptom_token_fatigue', 
          'symptom_symptom_token_feeling run-down or tired', 'symptom_symptom_token_fever', 'symptom_symptom_token_frequently waking', 
          'symptom_symptom_token_greenish cough', 'symptom_symptom_token_headache', 'symptom_symptom_token_heart palpitations', 'symptom_symptom_token_high fever', 
          'symptom_symptom_token_irritability', 'symptom_symptom_token_joint pain', 'symptom_symptom_token_loss of appetite', 
          'symptom_symptom_token_loss of appetite and unintentional weight loss', 'symptom_symptom_token_low energy', 'symptom_symptom_token_low-grade fever', 
          'symptom_symptom_token_lower back pain', 'symptom_symptom_token_morning headaches', 'symptom_symptom_token_mucus', 'symptom_symptom_token_muscle aches', 
          'symptom_symptom_token_nasal congestion', 'symptom_symptom_token_nausea', 'symptom_symptom_token_night sweats', 'symptom_symptom_token_pain', 
          'symptom_symptom_token_pauses in breathing', 'symptom_symptom_token_persistent dry coug', 'symptom_symptom_token_persistent dry cough', 
          'symptom_symptom_token_runny nose', 'symptom_symptom_token_shaking', 'symptom_symptom_token_shallow breathing', 'symptom_symptom_token_sharp chest pain', 
          'symptom_symptom_token_short', 'symptom_symptom_token_short of breath', 'symptom_symptom_token_shortness of breath', 
          'symptom_symptom_token_shortness of breath that gets worse during flare-ups', 'symptom_symptom_token_snoring', 'symptom_symptom_token_sore throat', 
          'symptom_symptom_token_stuffy nose', 'symptom_symptom_token_sweating', 'symptom_symptom_token_tight feeling in the chest', 
          'symptom_symptom_token_unusual moodiness', 'symptom_symptom_token_vomiting', 'symptom_symptom_token_weight loss', 
          'symptom_symptom_token_weight loss from loss of appetite', 'symptom_symptom_token_wheezing', 'symptom_symptom_token_wheezing cough', 
          'symptom_symptom_token_whistling sound while breathing', 'symptom_symptom_token_whistling sound while you breathe', 'symptom_symptom_token_yellow cough', 
          'sex_female', 'sex_male', 'sex_not to say', 'sex_unknown', 'nature_high', 'nature_low', 'nature_medium', 'nature_unknown', 
          'disease_acute respiratory distress syndrome', 'disease_asbestosis', 'disease_aspergillosis', 'disease_asthma', 'disease_bronchiectasis', 
          'disease_bronchiolitis', 'disease_bronchitis', 'disease_chronic bronchitis', 'disease_chronic cough', 'disease_chronic obstructive pulmonary disease', 
          'disease_influenza', 'disease_mesothelioma', 'disease_pneumonia', 'disease_pneumothorax', 'disease_pulmonary hypertension', 
          'disease_respiratory syncytial virus', 'disease_sleep apnea', 'disease_tuberculosis', 'age_group_child', 'age_group_teen', 'age_group_young_adult', 
          'age_group_adult', 'age_group_senior', 'symptom_count', 'high_risk']

app = Flask(__name__) #Create instance of flask app

#VALIDATION BLOCK
REQUIRED_FIELDS = ['age', 'symptom_1', 'symptom_2', 'symptom_3',
                   'sex', 'nature', 'age_group', 'symptom_count', 'high_risk']

def validate_input(data):

    #Checking all required fields are present
    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        return f"Missing required field(s): {', '.join(missing_fields)}"
    
    #Checking age is a positive integer
    try:
        age = float(data['age'])
    except (TypeError, ValueError):
        return "Age must be a numeric value."
    if age < 0:
        return "Age must be zero or greater."
    
    #Checking symptom count is a non-negative integer
    try:
        symptom_count = int(data['symptom_count'])
    except (TypeError, ValueError):
        return "Symptom count must be an integer."
    if symptom_count < 0:
        return "Symptom count must be zero or greater."
    
    #Check each symptom is NONE or known symptom
    valid_symptoms = set(data_process.symptoms)
    for field in ['symptom_1', 'symptom_2', 'symptom_3']:
        symptom = data[field]
        if symptom != "NONE" and symptom not in valid_symptoms:
            return f"Invalid value for {field}: '{symptom}'."
    
    #Checking sex, nature, age-gp are known values
    if data['sex'] not in data_process.gender:
        return f"Invalid value for sex: '{data['sex']}'."
    if data['nature'] not in data_process.nature:
        return f"Invalid value for nature: '{data['nature']}'."
    if data['age_group'] not in data_process.age_groups:
        return f"Invalid value for age_group: '{data['age_group']}'."
    
    #Checking high_risk is 'yes' or 'no' explicitly
    if not isinstance(data['high_risk'], str) or data['high_risk'] not in ('yes', 'no'):
        return f"Invalid value for high_risk: '{data['high_risk']}'. Must be 'yes' or 'no'."
    
    return None #No validation errors, input is valid

#VALIDATION BLOCK END

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*' 
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type' 
    return response


# TESTING CODE
# REASON: Adding a root path ('/') to confirm the server is running without getting a 404 error.
@app.route('/')
def health_check():
    return jsonify({
        "status": "online",
        "message": "Model API is running. Use /predict for POST requests."
    }), 200
# END OF TESTING CODE

@app.route('/predict', methods=['POST', 'OPTIONS']) #Declare post route called predict to predict the data
def add_item():

    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    #Get JSON data from the request body
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400
    
    validation_error = validate_input(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    value_list = [
        float(data['age']),
        data['symptom_1'],
        data['symptom_2'],
        data['symptom_3'],
        data['sex'],
        data['nature'],
        data['age_group'],
        int(data['symptom_count']),
        data['high_risk']
    ]
    
    numeric_values = data_process.data_processing(value_list) #Converts human readable string information into numeric data the model expects
        
    pred_arr = np.array([numeric_values]) #Convert the list into an array of one element that contains all values

    df_pred = pd.DataFrame(data=pred_arr, columns=cols)

    try:
        prediction = loaded_model.predict(df_pred) #Predict disease

        numeric_values[81:81] = data_process.multi_encode(prediction[0]) #Insert disease prediction for treatment recomendation

        pred_arr_2 = np.array([numeric_values]) #Convert the list into an array of one element that contains all values

        df_pred_2 = pd.DataFrame(data=pred_arr_2, columns=cols_2)

        prediction_2 = loaded_model_2.predict(df_pred_2) #Recomend Treatment

        res = {"Illness": prediction[0],
       "Treatment": prediction_2[0]}
        return jsonify(res), 201
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    


if __name__ == '__main__': #Launch the flask api with debugging set to true
    app.run(debug=True, use_reloader=False)
