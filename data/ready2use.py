from indices import get_monthly_returns
from indicators import get_indicators_change


def get_data(index, indicators, start, end, lag):
    df0 = get_monthly_returns(index, start, end)
    df1 = get_indicators_change(indicators, start, end)
    df = df1.iloc[:-lag]
    df[index] = df0.shift(lag).dropna()
    return df0, df1, df