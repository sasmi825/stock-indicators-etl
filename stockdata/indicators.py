import numpy as np
import pandas as pd
import talib
from pydantic import BaseModel

from .utils import WINDOW_DIVIDER, get_market_open_close_unix


def has_19_digits(array):
    """
    Checks if all elements in an array have 19 digits.

    Args:
        array: The array to check.

    Returns:
        True if all elements have 19 digits, False otherwise.
    """
    for element in array:
        if len(str(element)) != 19:
            return False
    return True


def check_increments_of_60(array):
    """
    Checks if elements in an array are in increments of 60.

    Args:
        array: The array to check.

    Returns:
        True if the elements are in increments of 60, False otherwise.
    """
    for i in range(1, len(array)):
        if array[i] - array[i - 1] != 60:
            return False
    return True


def get_subsequence_indices(array):
    """
    Finds subsequences of 60, 120, or 180 in a given sequence.

    Args:
        array: A list of integers.

    Returns:
        A list of tuples, where each tuple contains the start and end indices of a subsequence.
    """
    subsequence_indices = []  # output
    len_arr = len(array)  # length of array

    start_index = 0
    for idx in range(1, len_arr):
        if array[idx] not in [60, 120, 180]:
            end_index = idx
            if end_index - start_index > 1:
                subsequence_indices.append((start_index, end_index))
            start_index = idx
        if idx == len_arr - 1:
            end_index = len_arr
            if end_index - start_index > 1:
                subsequence_indices.append((start_index, end_index))

    return subsequence_indices


def missing_timestamps(arr):
    """
    Returns missing timestamps from an array to ensure consecutive
    values differ by 60 seconds in the og array.

    Args:
        arr: The input array.

    Returns:
        Array with elements missing.
    """
    result = []
    prev_value = arr[0]
    # result.append(prev_value)

    for value in arr[1:]:
        difference = value - prev_value
        if difference > 60:
            steps = int((difference // 60) - 1)
            for _ in range(steps):
                result.append(int(prev_value + 60))
                prev_value += 60
        # result.append(value)
        prev_value = value

    return result


class GenerateIndicatorsConfig(BaseModel):
    # ticker
    ticker: str
    # date for filtering pre & post market data
    date: str
    # Daily data or minute data
    interval: str = "1m"

    # column for timestamps
    time_column: str = "window_start"
    # column for adjusted close price
    close_column: str = "adj_close"
    # other column names
    high_col: str = "high"
    low_col: str = "low"
    close_un_adj_col: str = "close"
    vol_col: str = "volume"

    # rate of change ratio features
    # k in rate of change ratio = (curr_price-prev_price@k) / prev_price@k
    num_prev_rocp: int = 6
    # momentum features
    momentum_features: list[str] = [
        "rsi",
        "mfi",
        "ultosc",
        "cmo",
        "aroonosc",
        "macd_hist",
        "ppo",
        "sok",
        "sok_hist",
        "adx",
        "adx_hist",
    ]

    # whether to scale features
    scale: bool = True
    skip_na: bool = True

    # ultimate osc settings
    ultosc_timeperiod1: int = 7
    ultosc_timeperiod2: int = 14
    ultosc_timeperiod3: int = 28

    # Aroon osc
    aroonosc_timeperiod: int = 25


class GenerateIndicators:
    def __init__(self, config: GenerateIndicatorsConfig, df: pd.DataFrame):
        self.config = config
        self.df: pd.DataFrame = df

    def fill_na(self, fill_method="linear"):
        # collect indices where there are null values
        null_cols = []
        for col in self.df.columns:
            col_null_idx = self.df.loc[self.df[col].isna()].index.tolist()
            if len(col_null_idx) > 0:
                null_cols.append(col)

        # interpolate for null values
        for col in null_cols:
            self.df[col] = self.df[col].interpolate(
                method=fill_method, limit_direction="both", axis=0
            )

    def format_dataframe(self):
        """Formats df for the indicator generation"""
        self.df = self.df.sort_values(by=self.config.time_column, ascending=True)
        self.df = self.df.reset_index(drop=True)
        self.df[self.config.vol_col] = self.df[self.config.vol_col].astype(float)

    def filter_pre_post_market_data(self):
        """Filter out pre and post market data."""
        mst, met = get_market_open_close_unix(self.config.date)
        self.df = self.df.loc[
            (self.df.window_start >= mst) & (self.df.window_start < met)
        ]

    def break_dataframe(self):
        """Break dataframe to contain only 60, 120, or 180s time difference."""
        # get time difference between consecutive timestamps in seconds.
        ws = (self.df.window_start / WINDOW_DIVIDER).to_numpy()
        diff = talib.MOM(ws, timeperiod=1)
        # get indices for sub-sequences
        subsequence_indices = get_subsequence_indices(diff)

        sub_ticker_list = []
        df_list = []
        for i, indices in enumerate(subsequence_indices):
            dft_subseq = self.df.iloc[indices[0] : indices[1]]
            sub_tic = f"{self.config.ticker}-{i}"
            dft_subseq = dft_subseq.assign(ticker=[sub_tic] * len(dft_subseq))
            sub_ticker_list.append(sub_tic)
            df_list.append(dft_subseq)
        return sub_ticker_list, df_list

    def add_missing_timestamps(self):
        """Adds missing timestamps so that consecutive ts are 60s apart."""
        missing_ts = missing_timestamps(
            (self.df[self.config.time_column] / WINDOW_DIVIDER).tolist()
        )
        dft_missing = pd.DataFrame.from_dict(
            {
                "window_start": missing_ts,
                "ticker": [self.config.ticker] * len(missing_ts),
            }
        )
        dft_missing = dft_missing.assign(
            window_start=dft_missing.window_start * WINDOW_DIVIDER
        )
        # merge og and missing dfs
        self.df = pd.concat([self.df, dft_missing], ignore_index=True)

    def generate_indicators(self, skip_na=True) -> pd.DataFrame:
        if len(self.df) == 0:
            print(f"Empty dataframe was passed: {self.config.ticker}")
            return pd.DataFrame()

        # check if timestamps have 19 digits or not
        if not has_19_digits(self.df[self.config.time_column]):
            ValueError(
                f"Time column ('{self.config.time_column}') doesn't have 19 digits: {self.config.ticker}"
            )

        # Sort dataframe
        self.df = self.df.sort_values(by=self.config.time_column, ascending=True)

        # filter out pre & post market data
        self.filter_pre_post_market_data()
        if len(self.df) == 0:
            print(f"No market hours data for the ticker: {self.config.ticker}")
            return pd.DataFrame()

        # Break df into sub-dfs with only 60, 120, or 180 intervals diff in a given sequence.
        sub_ticker_list, df_list = self.break_dataframe()

        # for each sub-ticker generate indicators.
        df_indicator_list = []
        for sub_tic, df in zip(sub_ticker_list, df_list, strict=False):
            self.df = df
            df_tmp = self.per_ticker_run(sub_ticker=sub_tic, skip_na=skip_na)
            if len(df_tmp) > 0:
                df_indicator_list.append(df_tmp)

        # return dataframe
        if len(df_indicator_list) == 0:
            return pd.DataFrame()
        else:
            df = pd.concat(df_indicator_list, ignore_index=True)
            return df

    def per_ticker_run(self, sub_ticker: str, skip_na=True) -> pd.DataFrame:
        # Add missing timestamps
        self.add_missing_timestamps()
        # Format dataframe
        self.format_dataframe()
        if not check_increments_of_60(
            (self.df[self.config.time_column] / WINDOW_DIVIDER).tolist()
        ):
            ValueError(
                f"Time column ('{self.config.time_column}') is not in 60s increments: {self.config.ticker}"
            )

        # Fill null values in the dataframe
        self.fill_na()

        # Generate indicators
        fea_dict = {
            "window_start": self.df[self.config.time_column].to_numpy(),
            "close_price": self.df[self.config.close_column].to_numpy(),
        }

        for i in range(1, self.config.num_prev_rocp):
            fea_dict[f"rocp_{i}"] = self.get_rate_change_perc(timeperiod=i)

        # momentum indicators
        if "rsi" in self.config.momentum_features:
            fea_dict["rsi"] = self._get_rsi()
        if "mfi" in self.config.momentum_features:
            fea_dict["mfi"] = self._get_mfi()
        if "ultosc" in self.config.momentum_features:
            fea_dict["ultosc"] = self._get_ultimate_osc()
        if "cmo" in self.config.momentum_features:
            fea_dict["cmo"] = self._get_cmo()
        if "aroonosc" in self.config.momentum_features:
            fea_dict["aroonosc"] = self._get_aroon_osc()
        if "macd_hist" in self.config.momentum_features:
            fea_dict["macd_hist"] = self._get_macd()
        if "ppo" in self.config.momentum_features:
            fea_dict["ppo"] = self._get_ppo()
        if "sok" in self.config.momentum_features:
            sok_ = self._get_stoch_osc()
            fea_dict["sok"] = sok_[0]
            fea_dict["sok_hist"] = sok_[1]
        if "adx" in self.config.momentum_features:
            adx_ = self._get_adx()
            fea_dict["adx"] = adx_[1]
            fea_dict["adx_hist"] = adx_[0]

        df = pd.DataFrame.from_dict(fea_dict)
        df = df.assign(ticker=sub_ticker)
        if skip_na:
            return df.dropna()
        else:
            return df

    def _get_momentum(self, timeperiod: int = 10):
        """
        mom = curr_price - price@k  where k is kth prev period.
        """
        return talib.MOM(
            self.df[self.config.close_column].to_numpy(), timeperiod=timeperiod
        )  # 10

    def get_rate_change_perc(self, timeperiod: int = 1):  # scale=True
        """rate of change ration = curr_price - prev_price@k/prev_price@k"""
        rc = talib.ROCP(self.df[self.config.close_column], timeperiod=timeperiod)

        # if scale:
        #    return np.log(rc)
        # else:
        #    return rc
        return rc

    def _get_rsi(self, scale=True):
        """
        Relative Strength Index

        Trend
        """
        rsi = talib.RSI(self.df[self.config.close_column].to_numpy())
        if scale:
            return rsi / 100
        else:
            return rsi

    def _get_mfi(self, scale=True):
        """
        Money Flow Index

        Trend
        """
        mfi = talib.MFI(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
            self.df[self.config.vol_col].to_numpy(),
        )
        if scale:
            return mfi / 100
        else:
            return mfi

    def _get_ultimate_osc(self, scale=True):
        """
        Ultimate Oscillator

        Trend
        """
        ultosc = talib.ULTOSC(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
            timeperiod1=self.config.ultosc_timeperiod1,
            timeperiod2=self.config.ultosc_timeperiod2,
            timeperiod3=self.config.ultosc_timeperiod3,
        )
        if scale:
            return ultosc / 100
        else:
            return ultosc

    def _get_cci(self):
        """
        Commodity Channel Index

        Trend
        """
        # unbounded
        cci = talib.CCI(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
        )
        return cci

    def _get_stoch_osc(self, scale=True):
        """
        Stochastic Oscillator; returns histograms as well

        Trend
        """
        sok, sod = talib.STOCHF(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
        )
        so_hist = np.subtract(sok, sod)
        if scale:
            return [sok / 100, so_hist / 100]
        else:
            return [sok, so_hist]

    def _get_cmo(self, scale=True):
        """
        Chande Momentum Oscillator

        Trend
        """
        cmo = talib.CMO(self.df[self.config.close_column])
        if scale:
            return cmo / 100
        else:
            return cmo

    def _get_aroon_osc(self, scale=True):
        """
        Aroon Oscillator

        Trend and Trend Strength
        """
        aroon_osc = talib.AROONOSC(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            # default is 14 but this metric is typically calculated for 25 periods
            timeperiod=self.config.aroonosc_timeperiod,
        )
        if scale:
            return aroon_osc / 100
        else:
            return aroon_osc

    def _get_macd(self, scale=True):
        """
        MACD

        Trend
        """
        # Use MACSEXT to use different MAs
        macd, macd_signal, macd_hist = talib.MACDFIX(
            self.df[self.config.close_column].to_numpy()
        )
        if scale:
            return macd_hist / 10
        else:
            return macd_hist

    def _get_ppo(self, scale=True):
        """
        Percentage Price Oscillator

        Trend
        """
        ppo = talib.PPO(self.df[self.config.close_column].to_numpy())
        if scale:
            return ppo / 100
        else:
            return ppo

    def _get_adx(self, scale=True):
        """
        ADX

        Trend strength and direction.
        """
        adx = talib.ADX(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
        )

        pdi = talib.PLUS_DI(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
        )
        mdi = talib.MINUS_DI(
            self.df[self.config.high_col].to_numpy(),
            self.df[self.config.low_col].to_numpy(),
            self.df[self.config.close_un_adj_col].to_numpy(),
        )

        di_hist = np.subtract(pdi, mdi)

        if scale:
            return [di_hist / 100, adx / 100]
        else:
            return [di_hist, adx]
