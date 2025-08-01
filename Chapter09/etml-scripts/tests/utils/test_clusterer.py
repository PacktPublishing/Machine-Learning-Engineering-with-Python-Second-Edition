import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.cluster import Clusterer
from utils.extractor import Extractor
from tests.test_config import test_config

import datetime


def test_cluster_data():
    clusterer = Clusterer(test_config['bucket_name'], test_config['file_name'])
    clusterer.cluster_and_label(['ride_dist', 'ride_time'])
    date = datetime.datetime.now().strftime("%Y%m%d")
    extractor = Extractor(test_config['bucket_name'], f"clustered_data_{date}.json")
    df = extractor.extract_data()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df, pd.DataFrame)
    assert 'label' in df.columns
    assert df['label'].nunique() == 2
    assert -1 in df['label'].unique()
    assert 0 in df['label'].unique()