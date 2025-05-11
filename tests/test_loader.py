import pandas as pd
import pytest
from data.loader import fetch_spot_history, fetch_option_quotes, clean_option_quotes

@pytest.mark.parametrize("ticker,start,end", [
    ("SPY", "2021-01-01", "2021-01-05"),
    ("AAPL", "2022-06-01", "2022-06-03"),
])
def test_fetch_spot_history_not_empty(ticker, start, end):
    df = fetch_spot_history(ticker, start=start, end=end)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(['date','Open','Close','Volume']).issubset(df.columns)

def test_fetch_option_quotes_structure():
    df = fetch_option_quotes("SPY")
    expected = {'strike','bid','ask','mid_price','mid_iv','expiry','type'}
    assert isinstance(df, pd.DataFrame)
    assert expected.issubset(df.columns)

def test_clean_option_quotes_filters():
    df = fetch_option_quotes("SPY")
    df2 = clean_option_quotes(df, min_volume=0)  # skip volume filter
    # no NaNs
    assert not df2[['bid','ask','mid_price','mid_iv']].isnull().any().any()
    # bid < ask
    assert (df2['bid'] < df2['ask']).all()
    # iv in [0,5]
    assert df2['mid_iv'].between(0.0, 5.0).all()
