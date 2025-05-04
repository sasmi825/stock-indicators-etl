import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import dotenv_values
from tqdm import tqdm

from stockdata.utils import break_chunks, save_parquet_file

ENV_VARS = {
    **dotenv_values(
        "/home/diadochus/Desktop/repos/finance-starter-exp/stockdata/.env.1m"
    )
}
TickerDictType = dict[str, dict[str, list]]


def get_spy_tickers() -> list[str]:
    """Returns tickers in S&P index"""
    # read the file with SPY tickers information
    df = pd.read_csv(ENV_VARS["SPY_TICKER_FILE"])
    return df.Symbol.tolist() + ["SPY", "VOO"]


def get_ticker_dict(df: pd.DataFrame) -> TickerDictType:
    """Converts a multi-index dataframe to dict"""
    df_dict = {}
    for mul_col in df.columns:
        tic, col = mul_col

        if tic not in df_dict:
            df_dict[tic] = {}

        df_dict[tic][col] = df[mul_col].tolist()
    return df_dict


def get_formatted_df(df_dict: TickerDictType, window_start: list[int]) -> pd.DataFrame:
    """Converts dict of tickers data to dataframe"""
    req_cols = [
        "ticker",
        "volume",
        "open",
        "close",
        "high",
        "low",
        "adj_close",
        "window_start",
    ]

    df_list = []
    for tic in df_dict.keys():
        dft = pd.DataFrame.from_dict(df_dict[tic])
        dft = dft.assign(ticker=tic)
        dft = dft.assign(window_start=window_start)
        dft.columns = [col.lower().replace(" ", "_") for col in dft.columns]

        dft = dft[req_cols]
        df_list.append(dft)
    df = pd.concat(df_list, ignore_index=True)

    return df


def main(arg_parser):
    # ds, logical date
    airflow_date = arg_parser.execution_date
    date_format = "%Y-%m-%d"
    # get end date
    next_day = datetime.strptime(airflow_date, date_format) + timedelta(days=1)
    end_date = next_day.strftime(date_format)

    # interval
    interval = arg_parser.interval

    # get all SPY 500 tickers
    spy_tickers = get_spy_tickers()
    # break the list into chunks of size 10
    ticker_chunks = [
        x.tolist() for x in break_chunks(np.array(spy_tickers), chunk_size=10)
    ]

    # download all the ticker data
    df_list = []
    for tc in tqdm(ticker_chunks):
        df_raw = yf.download(
            tc,  # ["GOOGL", "SPY"],
            start=airflow_date,  # "%Y-%m-%d"
            end=end_date,  # exclusive
            interval=interval,
            group_by="ticker",
            prepost=False,
            progress=False,
        )
        # if no data was available skip
        if len(df_raw) == 0:
            continue
        window_start = df_raw.index.astype(np.int64)
        df_tc = get_formatted_df(get_ticker_dict(df_raw), window_start)
        df_list.append(df_tc)

    # if no data is downloaded skip
    if len(df_list) == 0:
        return 0
    df = pd.concat(df_list, ignore_index=True)

    # save file
    save_parquet_file(df, airflow_date, ENV_VARS["YAHOO_BASE"], interval)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution_date", type=str, help="Execution date received from Airflow"
    )
    parser.add_argument("--interval", type=str, help="Data interval to download 1m 1d")
    main(parser.parse_args())
