import pandas as pd


def calculate_statistics(df):
    return df.describe()

def make_func_result_json(func, df):
    return func(df).to_json()