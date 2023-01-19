import numpy
import pytest
import outliers.utils.data


@pytest.fixture()
def dummy_data():
    data = outliers.utils.data.create_data()
    return data

def test_data_is_numpy(dummy_data):
    assert isinstance(dummy_data, numpy.ndarray)

def test_data_is_large(dummy_data):
    assert len(dummy_data)>100
