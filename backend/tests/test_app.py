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