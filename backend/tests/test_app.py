"""Script for running unit tests for app."""

import json
import pytest
from tests.fixtures import VALID_PAYLOAD

#Health check test -- API-01
def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'online'


#Test for valid payload -- API-02
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



#Test for missing fields in payload -- API-03
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



#Test for invalid 'high_risk' value -- API-04
def test_invalid_high_risk_value_returns_400(client):
    payload = {**VALID_PAYLOAD, 'high_risk': 1} # Invalid value, should be 'yes' / 'no'

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400


#Test for invalid symptom value reteruns 400 -- API-05
def test_invalid_symptom_returns_400(client):
    payload = {**VALID_PAYLOAD, 'symptom_1': 'fake_symp_test'}

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400
    assert 'symptom_1' in response.get_json()['error']


#Test for symptom count out of range -- API-06
def test_symptom_count_out_of_range_returns_400(client):
    payload = {**VALID_PAYLOAD, 'symptom_count': -1}

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400


#Test for negative age value returns 400 -- API-07
def test_negative_age_returns_400(client):
    payload = {**VALID_PAYLOAD, 'age': -5}

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 400


#Test for non-json payload retruns 415 -- API-08
def test_non_json_payload_returns_400(client):
    response = client.post(
        '/predict',
        data="This is not JSON",
        content_type='text/plain'
    )
    assert response.status_code == 415



#Test for GET request to /predict returns 405 -- API-09
def test_get_request_to_predict_returns_405(client):
    response = client.get('/predict')
    assert response.status_code == 405


#Test for model exception handling returns 500 -- API-10
def test_model_exception_handling_returns_500(client, illness_model):
    illness_model.predict.side_effect = Exception("Model error")

    response = client.post(
        '/predict',
        data=json.dumps(VALID_PAYLOAD),
        content_type='application/json'
    )
    assert response.status_code == 500
    assert 'Error' in response.get_json()