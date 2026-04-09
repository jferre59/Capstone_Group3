import joblib
import numpy as np
import pandas as pd
import os


symptoms = ['cough that lasts more than three weeks',
       'dry', 'allergy',
       'bluish skin', 'breath',
       'chest congestion',
       'chest pain', 'chills',
       'chronic cough', 'cold',
       'cough', 'cough with blood',
       'coughing',
       'coughing up blood',
       'coughing up yellow or green mucus daily',
       'crackling sound in the lungs while breathing in',
       'daytime sleepiness',
       'diarrhea',
       'difficulties with memory and concentration',
       'distressing', 'dizziness',
       'dry cough', 'dry mouth',
       'edema', 'token_fainting',
       'faster heart beating',
       'fatigue',
       'feeling run-down or tired',
       'fever',
       'frequently waking',
       'greenish cough',
       'headache',
       'heart palpitations',
       'high fever',
       'irritability',
       'joint pain',
       'loss of appetite',
       'loss of appetite and unintentional weight loss',
       'low energy',
       'low-grade fever',
       'lower back pain',
       'morning headaches',
       'mucus', 'muscle aches',
       'nasal congestion',
       'nausea', 'night sweats',
       'token_pain',
       'pauses in breathing',
       'persistent dry coug',
       'persistent dry cough',
       'runny nose', 'shaking',
       'shallow breathing',
       'sharp chest pain', 'short',
       'short of breath',
       'shortness of breath',
       'shortness of breath that gets worse during flare-ups',
       'snoring', 'sore throat',
       'stuffy nose', 'sweating',
       'tight feeling in the chest',
       'unusual moodiness',
       'vomiting', 'weight loss',
       'weight loss from loss of appetite',
       'wheezing',
       'wheezing cough',
       'whistling sound while breathing',
       'whistling sound while you breathe',
       'yellow cough']

gender = ['female', 'male',
       'not to say', 'unknown']

nature = ['high', 'low',
       'medium', 'unknown']

age_gp = ['child', 'teen',
       'young adult', 'adult', 'senior']

disease = ['acute respiratory distress syndrome', 'asbestosis',
       'aspergillosis', 'asthma', 'bronchiectasis',
       'bronchiolitis', 'bronchitis',
       'chronic bronchitis', 'chronic cough',
       'chronic obstructive pulmonary disease', 'influenza',
       'mesothelioma', 'pneumonia', 'pneumothorax',
       'pulmonary hypertension', 'respiratory syncytial virus',
       'sleep apnea', 'tuberculosis']

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
output_model = os.path.join(backend_dir, 'model', 'scaler.joblib')
os.makedirs(os.path.dirname(output_model), exist_ok=True)
loaded_scaler = joblib.load(output_model)

def data_processing(data):
    output = []
    scale = np.array([[data[0], data[5]]])
    scaled = loaded_scaler.transform(scale) #Scales age and symptom count for use in data


    output.append(scaled[0][0].item())

    symp = []
    for s in range(0, 73):
        symp.append(0)

    if data[1] != "NONE":
        l = symptoms.index(data[1])
        symp[l] = 1

    output.extend(symp)

    sex = [0,0,0,0]
    i = gender.index(data[2])
    sex[i] = 1
    output.extend(sex)

    nature_bin = [0,0,0,0]
    x = nature.index(data[3])
    nature_bin[x] = 1
    output.extend(nature_bin)

    age_group = [0,0,0,0,0]
    z = age_gp.index(data[4])
    age_group[z] = 1
    output.extend(age_group)

    output.append(scaled[0][1].item())

    if data[6] == 'yes':
        output.append(1)
    else:
        output.append(0)

    return output

def multi_encode(illness):
    ill = []
    for s in range(0, 18):
        ill.append(0)

    i = disease.index(illness)
    ill[i] = 1
    return ill

'''
nature_s = ['age', 'symptom_symptom_token_a cough that lasts more than three weeks', 'symptom_symptom_token_a dry', 
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
            'sex_female', 'sex_male', 'sex_not to say', 'sex_unknown', 'nature_high', 'nature_low', 'nature_medium', 'nature_unknown']

print(nature_s.index('nature_unknown'))
'''
