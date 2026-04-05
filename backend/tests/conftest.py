"""Script to configure test environment. Run before tests."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

#Patching scaler and models before importing app as models load at module level
mock_scaler = MagicMock()
mock_illness_model = MagicMock()
mock_treatment_model = MagicMock()

#Transforming mock_scaler
mock_scaler.transform.return_value = np.array([[0.5, 0.5]])

with patch('joblib.load', side_effect=[mock_scaler, mock_illness_model, mock_treatment_model]):
    from services import app as flask_app

@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as client:
        yield client

@pytest.fixture
def illness_model():
    return mock_illness_model

@pytest.fixture
def treatment_model():
    return mock_treatment_model