import pandas as pd
import yfinance as yf


def fetch_spot_history(
    ticker: str,
    start: str = "2000-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Download daily spot price history for given ticker and clean columns.

    Parameters
    ----------
    ticker : str
        Equity symbol, e.g. 'SPY'
    start : str
        Start date 'YYYY-MM-DD'
    end : str, optional
        End date 'YYYY-MM-DD' (defaults to today)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['date','Open','High','Low','Close','Volume']
    """
    df = yf.download(
        ticker,
        start=start,
        end=end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        progress=False
    )
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    # flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # keep only needed columns
    cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df.loc[:, cols]


def fetch_option_quotes(
    ticker: str,
) -> pd.DataFrame:
    """
    Download option chain quotes for all expiries via yfinance.

    Parameters
    ----------
    ticker : str
        Equity symbol

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['contractSymbol','lastPrice','bid','ask','impliedVolatility',
         'inTheMoney','contractSize','currency','type','strike','expiry']
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    records = []

    for expiry in expiries:
        chain = tk.option_chain(expiry)
        for kind in ['calls', 'puts']:
            df = getattr(chain, kind).copy()
            df['type'] = kind[:-1]
            df['expiry'] = pd.to_datetime(expiry)
            records.append(df)

    df_opts = pd.concat(records, ignore_index=True)
    # mid implied vol directly from yfinance
    df_opts.rename(columns={'impliedVolatility': 'mid_iv'}, inplace=True)
    # mid price from bid/ask
    df_opts['mid_price'] = (df_opts['bid'] + df_opts['ask']) / 2
    return df_opts[['strike', 'bid', 'ask', 'mid_price', 'mid_iv', 'expiry', 'type']]


def clean_option_quotes(
    df: pd.DataFrame,
    min_volume: int = 100,
    iv_range: tuple = (0.0, 5.0),
) -> pd.DataFrame:
    """
    Filter out stale or illiquid option quotes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw option quotes DataFrame
    min_volume : int
        Minimum volume threshold
    iv_range : tuple
        Allowable implied volatility range (low, high)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with columns:
        ['strike','bid','ask','mid_price','mid_iv','expiry','type']
    """
    df = df.dropna(subset=['bid', 'ask', 'mid_price', 'mid_iv'])
    if 'volume' in df.columns:
        df = df[df['volume'] >= min_volume]
    df = df[(df['mid_iv'] >= iv_range[0]) & (df['mid_iv'] <= iv_range[1])]
    df = df[df['bid'] < df['ask']]
    return df[['strike', 'bid', 'ask', 'mid_price', 'mid_iv', 'expiry', 'type']]
