import pytest
import pandas as pd
from khronos import transform

@pytest.fixture
def raw_sp500_aapl_and_msft_2018_dataframe():
    dataset = pd.read_csv('./artifacts/test/raw.csv', index_col=0, header=[0, 1]).sort_index(axis=1)
    return dataset

@pytest.fixture
def flat_df():
    data = {'ticker': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
            'datetime': ['2018-01-02 09:30:00', '2018-01-02 09:30:00', '2018-01-02 09:30:00',
                            '2018-01-02 09:30:00', '2018-01-02 09:30:00', '2018-01-02 09:30:00'],
            'Open': [170.16, 170.16, 170.16, 86.13, 86.13, 86.13],
            'High': [170.16, 170.16, 170.16, 86.13, 86.13, 86.13],
            'Low': [170.16, 170.16, 170.16, 86.13, 86.13, 86.13],
            'Close': [170.16, 170.16, 170.16, 86.13, 86.13, 86.13],
            'Volume': [0, 0, 0, 0, 0, 0]}
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index(['datetime'], inplace=True)

    return df

def test_flatten_dataframe_to_tick_per_row(raw_sp500_aapl_and_msft_2018_dataframe):
    flat_df = transform.flatten_dataset(raw_sp500_aapl_and_msft_2018_dataframe)
    assert 'ticker' in flat_df.columns
    assert len(flat_df.index.names) == 1

def test_transform_dataframe_to_tick_per_row(raw_sp500_aapl_and_msft_2018_dataframe):
    flat_df = transform.flatten_dataset(raw_sp500_aapl_and_msft_2018_dataframe)
    tick_per_row = transform.to_multi_index_df(flat_df)
    # tick_per_row = transform.transform_dataset_to_tick_per_row(raw_sp500_aapl_and_msft_2018_dataframe)
    
    # select AAPL and datetime is between 2018-01-02 09:30:00 and 2018-01-02 16:00:00
    assert tick_per_row.loc['AAPL', '2018-01-02 09:30:00':'2018-01-02 09:32:00', :].shape == (3, 5)

    # assert msft and aapl are in the index
    assert 'AAPL' in tick_per_row.index
    assert 'MSFT' in tick_per_row.index
    
def test_add_timestamp_in_seconds_as_raw_input(flat_df):
    df = transform.add_timestamp_in_seconds_as_raw_input(flat_df)

    # convert seconds to datetime and check if it is equal to the datetime index
    seconds = df['timestamp_in_seconds'].apply(lambda x: pd.to_datetime(x, unit='s'))
    # assert all true
    assert (seconds == df.index).all()

