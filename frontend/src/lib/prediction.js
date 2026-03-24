export const symptomOptions = [
  { label: 'None', value: 'NONE' },
  { label: 'A cough that lasts more than three weeks', value: 'cough that lasts more than three weeks' },
  { label: 'Dry', value: 'dry' },
  { label: 'Allergy', value: 'allergy' },
  { label: 'Bluish skin', value: 'bluish skin' },
  { label: 'Breath', value: 'breath' },
  { label: 'Chest congestion', value: 'chest congestion' },
  { label: 'Chest pain', value: 'chest pain' },
  { label: 'Chills', value: 'chills' },
  { label: 'Chronic cough', value: 'chronic cough' },
  { label: 'Cold', value: 'cold' },
  { label: 'Cough', value: 'cough' },
  { label: 'Cough with blood', value: 'cough with blood' },
  { label: 'Coughing', value: 'coughing' },
  { label: 'Coughing up blood', value: 'coughing up blood' },
  { label: 'Coughing up yellow or green mucus daily', value: 'coughing up yellow or green mucus daily' },
  { label: 'Crackling sound in the lungs while breathing in', value: 'crackling sound in the lungs while breathing in' },
  { label: 'Daytime sleepiness', value: 'daytime sleepiness' },
  { label: 'Diarrhea', value: 'diarrhea' },
  { label: 'Difficulties with memory and concentration', value: 'difficulties with memory and concentration' },
  { label: 'Distressing', value: 'distressing' },
  { label: 'Dizziness', value: 'dizziness' },
  { label: 'Dry cough', value: 'dry cough' },
  { label: 'Dry mouth', value: 'dry mouth' },
  { label: 'Edema', value: 'edema' },
  { label: 'Faster heart beating', value: 'faster heart beating' },
  { label: 'Fatigue', value: 'fatigue' },
  { label: 'Feeling run-down or tired', value: 'feeling run-down or tired' },
  { label: 'Fever', value: 'fever' },
  { label: 'Frequently waking', value: 'frequently waking' },
  { label: 'Greenish cough', value: 'greenish cough' },
  { label: 'Headache', value: 'headache' },
  { label: 'Heart palpitations', value: 'heart palpitations' },
  { label: 'High fever', value: 'high fever' },
  { label: 'Irritability', value: 'irritability' },
  { label: 'Joint pain', value: 'joint pain' },
  { label: 'Loss of appetite', value: 'loss of appetite' },
  { label: 'Loss of appetite and unintentional weight loss', value: 'loss of appetite and unintentional weight loss' },
  { label: 'Low energy', value: 'low energy' },
  { label: 'Low-grade fever', value: 'low-grade fever' },
  { label: 'Lower back pain', value: 'lower back pain' },
  { label: 'Morning headaches', value: 'morning headaches' },
  { label: 'Mucus', value: 'mucus' },
  { label: 'Muscle aches', value: 'muscle aches' },
  { label: 'Nasal congestion', value: 'nasal congestion' },
  { label: 'Nausea', value: 'nausea' },
  { label: 'Night sweats', value: 'night sweats' },
  { label: 'Pauses in breathing', value: 'pauses in breathing' },
  { label: 'Persistent dry coug', value: 'persistent dry coug' },
  { label: 'Persistent dry cough', value: 'persistent dry cough' },
  { label: 'Runny nose', value: 'runny nose' },
  { label: 'Shaking', value: 'shaking' },
  { label: 'Shallow breathing', value: 'shallow breathing' },
  { label: 'Sharp chest pain', value: 'sharp chest pain' },
  { label: 'Short', value: 'short' },
  { label: 'Short of breath', value: 'short of breath' },
  { label: 'Shortness of breath', value: 'shortness of breath' },
  { label: 'Shortness of breath that gets worse during flare-ups', value: 'shortness of breath that gets worse during flare-ups' },
  { label: 'Snoring', value: 'snoring' },
  { label: 'Sore throat', value: 'sore throat' },
  { label: 'Stuffy nose', value: 'stuffy nose' },
  { label: 'Sweating', value: 'sweating' },
  { label: 'Tight feeling in the chest', value: 'tight feeling in the chest' },
  { label: 'Unusual moodiness', value: 'unusual moodiness' },
  { label: 'Vomiting', value: 'vomiting' },
  { label: 'Weight loss', value: 'weight loss' },
  { label: 'Weight loss from loss of appetite', value: 'weight loss from loss of appetite' },
  { label: 'Wheezing', value: 'wheezing' },
  { label: 'Wheezing cough', value: 'wheezing cough' },
  { label: 'Whistling sound while breathing', value: 'whistling sound while breathing' },
  { label: 'Whistling sound while you breathe', value: 'whistling sound while you breathe' },
  { label: 'Yellow cough', value: 'yellow cough' },
];

export const sexOptions = [
  { label: 'Male', value: 'male' },
  { label: 'Female', value: 'female' },
  { label: 'Prefer not to say', value: 'not to say' },
  { label: 'Unknown', value: 'unknown' },
];

export const natureOptions = [
  { label: 'Low', value: 'low' },
  { label: 'Medium', value: 'medium' },
  { label: 'High', value: 'high' },
  { label: 'Unknown', value: 'unknown' },
];

export function deriveAgeGroup(ageValue) {
  const age = Number(ageValue);

  if (Number.isNaN(age) || age < 0) {
    return '';
  }

  if (age < 12) {
    return 'child';
  }

  if (age < 18) {
    return 'teen';
  }

  if (age < 35) {
    return 'young adult';
  }

  if (age < 65) {
    return 'adult';
  }

  return 'senior';
}

export function deriveHighRisk(ageValue, nature) {
  const age = Number(ageValue);

  if (Number.isNaN(age)) {
    return 'no';
  }

  return age < 12 || age > 65 ? (nature === 'high' ? 'yes' : 'no') : 'no';
}

export function getSelectedSymptoms(form) {
  return [form.symptom_1, form.symptom_2, form.symptom_3].filter((value) => value && value !== 'NONE');
}

export function buildPredictionPayload(form) {
  const selectedSymptoms = getSelectedSymptoms(form);

  return {
    age: Number(form.age),
    symptom_1: form.symptom_1 || 'NONE',
    symptom_2: form.symptom_2 || 'NONE',
    symptom_3: form.symptom_3 || 'NONE',
    sex: form.sex,
    nature: form.nature,
    age_group: deriveAgeGroup(form.age),
    symptom_count: selectedSymptoms.length,
    high_risk: deriveHighRisk(form.age, form.nature),
  };
}

export async function predictCondition(form) {
  const payload = buildPredictionPayload(form);
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:5000';

  const response = await fetch(`${apiBaseUrl}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const responseText = await response.text();
  const data = responseText ? JSON.parse(responseText) : null;

  if (!response.ok) {
    throw new Error(data?.error || data?.Error || 'Prediction request failed.');
  }

  return { payload, result: data };
}
