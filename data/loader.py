import pandas as pd
import yfinance as yf


def fetch_spot_history(
    ticker: str,
    start: str = "2000-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Download daily spot price history for given ticker.

    Parameters
    ----------
    ticker : equity symbol, e.g. 'SPY'
    start  : start date 'YYYY-MM-DD'
    end    : end date 'YYYY-MM-DD' (defaults to today)

    Returns
    -------
    DataFrame with columns: ['Date','Open','High','Low','Close','Adj Close','Volume']
    """
    df = yf.download(
        ticker,
        start=start,
        end=end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        progress=False
    )
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    return df


def fetch_option_quotes(
    ticker: str,
) -> pd.DataFrame:
    """
    Download option chain quotes for all expiries via yfinance.
    Returns bid, ask, implied vol for each contract.

    Parameters
    ----------
    ticker : equity symbol

    Returns
    -------
    DataFrame with columns: [
        'contractSymbol','lastPrice','bid','ask','impliedVolatility',
        'inTheMoney','contractSize','currency','type','strike','expiry'
    ]
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
    return df_opts


def clean_option_quotes(
    df: pd.DataFrame,
    min_volume: int = 100,
    iv_range: tuple = (0.0, 5.0),
) -> pd.DataFrame:
    """
    Filter out stale or illiquid option quotes.

    Parameters
    ----------
    df : raw option quotes DataFrame
    min_volume : minimum volume threshold
    iv_range : allowable implied volatility range

    Returns
    -------
    Filtered DataFrame
    """
    # drop NaNs
    df = df.dropna(subset=['bid', 'ask', 'mid_iv'])
    # volume filter if column exists
    if 'volume' in df.columns:
        df = df[df['volume'] >= min_volume]
    # implied vol range
    df = df[(df['mid_iv'] >= iv_range[0]) & (df['mid_iv'] <= iv_range[1])]
    # bid<ask
    df = df[df['bid'] < df['ask']]
    return df
