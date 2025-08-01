'''
Unit tests for the extractor module.
'''
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.extractor import Extractor
from tests.test_config import test_config

def test_extract_data():
    extractor = Extractor(test_config['bucket_name'], test_config['file_name'])
    df = extractor.extract_data()
    assert isinstance(df, pd.DataFrame)
