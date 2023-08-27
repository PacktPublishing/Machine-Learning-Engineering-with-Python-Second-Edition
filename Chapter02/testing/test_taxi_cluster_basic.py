import pytest
from model import cluster_and_label
from helper import get_taxi_data

@pytest.mark.skip(reason="From edition 1, does not work due to not uploading taxi data in repo")
def test_cluster_and_label():
    df = get_taxi_data()
    results = cluster_and_label(df)
    assert isinstance(results, dict)
