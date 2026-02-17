import numpy as np

symptoms = [(13, 'coughing'), (66, 'tight feeling in the chest'), (71, 'wheezing'), (60, 'shortness of breath'), (29, 'fever'), 
        (10, 'cold'), (2, 'allergy'), (15, 'coughing up yellow or green mucus daily'), (61, 'shortness of breath that gets worse during flare-ups'), 
        (27, 'fatigue, feeling run-down or tired'), (6, 'chest pain'), (74, 'whistling sound while you breathe'), (14, 'coughing up blood'), 
        (54, 'runny nose'), (64, 'stuffy nose'), (37, 'loss of appetite'), (11, 'cough'), (40, 'low-grade fever'), (5, 'chest congestion'), 
        (73, 'whistling sound while breathing'), (76, 'yellow cough'), (28, 'feeling run-down or tired'), (43, 'mucus'), (9, 'chronic cough'), 
        (26, 'fatigue'), (41, 'lower back pain'), (21, 'dry cough'), (31, 'greenish cough'), (12, 'cough with blood'), (65, 'sweating'), 
        (55, 'shaking'), (52, 'rapid breathing'), (56, 'shallow breathing'), (39, 'low energy'), (46, 'nausea'), (68, 'vomiting'), (57, 'sharp chest pain'), 
        (3, 'bluish skin'), (53, 'rapid heartbeat'), (34, 'high fever'), (32, 'headache'), (44, 'muscle aches'), (36, 'joint pain'), (8, 'chills'), 
        (63, 'sore throat'), (45, 'nasal congestion'), (17, 'diarrhea'), (4, 'breath'), (20, 'dizziness'), (24, 'fainting'), (33, 'heart palpitations'), 
        (23, 'edema'), (62, 'snoring'), (16, 'daytime sleepiness'), (49, 'pauses in breathing'), (18, 'difficulties with memory and concentration'), 
        (67, 'unusual moodiness'), (35, 'irritability'), (30, 'frequently waking'), (42, 'morning headaches'), (22, 'dry mouth'), (72, 'wheezing cough'), 
        (59, 'short, shallow and rapid breathing'), (0, 'a cough that lasts more than three weeks'), (38, 'loss of appetite and unintentional weight loss'), 
        (47, 'night sweats'), (58, 'short of breath'), (19, 'distressing'), (25, 'faster heart beating'), (48, 'pain'), (51, 'persistent dry cough'), 
        (7, 'chest tightness or chest pain'), (70, 'weight loss from loss of appetite'), (1, 'a dry, crackling sound in the lungs while breathing in'), 
        (75, 'wider and rounder than normal fingertips and toes'), (50, 'persistent dry coug'), (69, 'weight loss')]



def data_processing(data):
    output = []
    if data[0] == 'male':
        output.append(0)
    else:
        output.append(1)

    if data[1] == 'high':
        output.append(2)
    elif data[1] == 'medium':
        output.append(1)
    else:
        output.append(0)

    if data[2] == 'child':
        output.append(1)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
    elif data[2] == 'teen':
        output.append(0)
        output.append(1)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
    elif data[2] == 'young adult':
        output.append(0)
        output.append(0)
        output.append(1)
        output.append(0)
        output.append(0)
        output.append(0)
    elif data[2] == 'adult':
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(1)
        output.append(0)
        output.append(0)
    elif data[2] == 'middle aged':
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(1)
        output.append(0)
    else:
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(0)
        output.append(1)

    if data[3] == 'yes':
        output.append(1)
    else:
        output.append(0)

    search_value = data[4]
    try:
        index = next(i for i, tup in enumerate(symptoms) if search_value in tup)
        output.append(symptoms[index][0])
    except StopIteration:
        print(f"'{search_value}' not found in any tuple.")

    return output