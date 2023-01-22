import logging
import pandas as pd


'''
Basic logging section
'''
logging.basicConfig(filename='advanced_logging.log',
                    level=logging.DEBUG,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

logging.debug('Message to help debug ...')
logging.info('General info about a process that is running ...')
logging.warning('Warn, but no need to error ...')


def feature_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a dataframe by not doing anything. Just for demo.

    :param df: a dataframe.
    :return: df.mean(), the same dataframe with the averages of each column.
    """
    return df.mean()

df = pd.DataFrame(data={'col1': [1,2,3,4], 'col2': [5,6,7,8]})

try:
    df_transformed = feature_transform(df)
    logging.info("df successfully transformed")
except Exception as err:
    logging.error("Unexpected error", exc_info=True)


list_of_nums = [1,2,3,4,5,6,7,8]

try:
    df_transformed = feature_transform(list_of_nums)
    logging.info("df successfully transformed")
except Exception as err:
    logging.error("Unexpected error", exc_info=True)
