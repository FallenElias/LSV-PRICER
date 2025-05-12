import pandas as pd
import yfinance as yf
import datetime as dt


def fetch_spot_history(
    ticker: str,
    years: int = 3,
) -> pd.DataFrame:
    """
    Download the last N years of daily closing prices for `ticker`.

    Parameters
    ----------
    ticker : str
        Equity symbol, e.g., 'AAPL'
    years  : int
        Number of years of history to fetch (default 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date','Close']
    """
    end   = dt.date.today()
    start = end - dt.timedelta(days=years * 365)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,  # preserves raw Close
    )
    df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'date'})
    return df

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
