import pandas as pd
import yfinance as yf
import datetime as dt
from utils.financial import bs_implied_vol


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
    S0: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    """
    Download option chain from Yahoo, compute mid_price (bid/ask midpoint),
    drop any bad quotes, then invert to mid_iv via bs_implied_vol.
    """
    tk       = yf.Ticker(ticker)
    expiries = tk.options
    records  = []

    for expiry in expiries:
        # time to expiry in years
        T     = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365.0
        chain = tk.option_chain(expiry)

        for kind in ('calls','puts'):
            df = getattr(chain, kind).copy()
            df['type']   = kind[:-1]
            df['expiry'] = pd.to_datetime(expiry)
            df['T']      = T

            # mid‐price and basic sanity filters
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df = df.dropna(subset=['bid','ask'])
            df = df[df['bid'] < df['ask']]
            df = df[df['mid_price'] > 0]

            # invert to implied vol
            df['mid_iv'] = df.apply(
                lambda row: bs_implied_vol(
                    S0, row['strike'], row['T'], r, q, row['mid_price']
                ),
                axis=1
            )

            records.append(df)

    out = pd.concat(records, ignore_index=True)
    return out[[
        'strike','bid','ask','mid_price','mid_iv',
        'expiry','T','type'
    ]]


def clean_option_quotes(
    df: pd.DataFrame,
    min_volume: int = 100,
    iv_range: tuple = (0.0, 5.0),
) -> pd.DataFrame:
    """
    Keep only liquid, in‐range call quotes.
    """
    df = df.dropna(subset=['mid_price','mid_iv'])
    # keep calls only
    df = df[df['type'] == 'call']
    if 'volume' in df.columns:
        df = df[df['volume'] >= min_volume]
    df = df[(df['mid_iv'] >= iv_range[0]) & (df['mid_iv'] <= iv_range[1])]
    df = df[df['bid'] < df['ask']]
    return df[['strike','bid','ask','mid_price','mid_iv','expiry','T','type']]
