import pytest
from outliers.detectors.detection_models import DetectionModels
from outliers.detectors.pipelines import OutlierDetector
from outliers.definitions import MODEL_CONFIG_PATH
import outliers.utils.data
import numpy as np

@pytest.fixture()
def dummy_data():
    data = outliers.utils.data.create_data()
    return data

@pytest.fixture()
def example_models():
    models = DetectionModels(MODEL_CONFIG_PATH)
    return models

@pytest.fixture()
def example_detector(example_models):
    model = example_models.get_models()[0]
    detector = OutlierDetector(model=model)
    return detector

def test_model_creation(example_models):
    assert example_models is not None

def test_model_get_models(example_models):
    example_models.get_models() is not None

def test_model_evaluation(dummy_data, example_detector):
    result = example_detector.detect(dummy_data)
    assert len(result[result == -1]) == 39 #number of anomalies to detect
    assert len(result) == len(dummy_data) #same numbers of results
    assert np.unique(result)[0] == -1 #np.array([-1, 1])
    assert np.unique(result)[1] == 1  # np.array([-1, 1])
