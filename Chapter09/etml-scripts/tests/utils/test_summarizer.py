import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.summarize import LLMSummarizer
from utils.extractor import Extractor
from tests.test_config import test_config

import datetime
date = datetime.datetime.now().strftime("%Y%m%d")


def test_summarize_data():
    summarizer = LLMSummarizer(test_config['bucket_name'], f"clustered_data_{date}.json")
    summarizer.summarize()
    extractor = Extractor(test_config['bucket_name'], f"clustered_summarized_{date}.json")
    df = extractor.extract_data()
    assert isinstance(df, pd.DataFrame)
    assert 'summary' in df.columns
    assert df['summary'].type() == str