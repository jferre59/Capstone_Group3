"""Script for running unit tests for app."""

import json
import pytest
from tests.fixtures import VALID_PAYLOAD

#Health check test
def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'online'


#Test for valid payload
def test_valid_payload_returns_prediction(client, illness_model, treatment_model):
    illness_model.predict.return_value = ['pneumonia']
    treatment_model.predict.return_value = ['antibiotics']

    response = client.post(
        '/predict',
        data=json.dumps(VALID_PAYLOAD),
        content_type='application/json'
    )
    assert response.status_code == 201
    data = response.get_json()
    assert data['Illness'] == 'pneumonia'
    assert data['Treatment'] == 'antibiotics'



#Test for missing fields in payload
@pytest.mark.parametrize("missing_field", [
    'age', 'symptom_1', 'sex', 'nature', 'age_group', 'symptom_count', 'high_risk'
])
def test_missing_field_returns_400(client, missing_field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400
    assert 'error' in response.get_json()



#Test for invalid 'high_risk' value
def test_invalid_high_risk_value_returns_400(client):
    payload = {**VALID_PAYLOAD, 'high_risk': 1} # Invalid value, should be 'yes' / 'no'

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400