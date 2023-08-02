import pandas as pd
import numpy as np

def transform_dataset_to_tick_per_row(df: pd.DataFrame):
    # df is a multi-colum dataframe with the following structure:
    #   - index: datetime
    #   - columns: multi-index with levels:
    #       - level 0: ticker   (e.g. 'AAPL', 'MSFT')
    #       - level 1: feature  (e.g. 'Open', 'High', 'Low', 'Close', 'Volume')

    # We want to transform this dataframe into a single-column dataframe with the following structure:
    # - index: multi-index with levels:
    #       - level 0: ticker   (e.g. 'AAPL', 'MSFT')
    #       - level 1: datetime

    # - columns:
    #       - 'Open'
    #       - 'High'
    #       - 'Low'
    #       - 'Close'
    #       - 'Volume'

    # We can achieve this by:
    ticker_on_rows_df = df.stack(level=0)
    # set timestamp to datetime
    ticker_on_rows_df.index = ticker_on_rows_df.index.set_levels(pd.to_datetime(ticker_on_rows_df.index.levels[0]), level=0)
    # swap levels
    ticker_at_level_0_df = ticker_on_rows_df.swaplevel().sort_index()
    return ticker_at_level_0_df

def flatten_dataset(df: pd.DataFrame):
    date_and_ticker_index_df = df.stack(level=0)
    date_and_ticker_index_df['ticker'] = date_and_ticker_index_df.index.get_level_values(1)
    # drop the ticker index
    date_and_ticker_index_df.index = date_and_ticker_index_df.index.droplevel(1)
    return date_and_ticker_index_df

def convert_index_to_datetime(df: pd.DataFrame):
    df.index = pd.to_datetime(df.index)
    return df

def add_timestamp_in_seconds_as_raw_input(df: pd.DataFrame):
    # get index values
    datetime_values = df.index.values
    # convert to seconds
    timestamp_in_seconds = datetime_values.astype(np.int64) // 10**9
    # add as input call 'timestamp_in_seconds'
    df['timestamp_in_seconds'] = timestamp_in_seconds
    return df


def to_multi_index_df(df: pd.DataFrame):
    # add ticker as index level 0 and datetime as index level 1
    date_and_ticker_index_df = df.set_index(['ticker'], append=True)
    date_and_ticker_index_df.index = date_and_ticker_index_df.index.swaplevel()
    date_and_ticker_index_df.sort_index(inplace=True)
    return date_and_ticker_index_df

