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