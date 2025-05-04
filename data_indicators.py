import argparse
import os.path

import pandas as pd
from dotenv import dotenv_values

from stockdata.indicators import (
    GenerateIndicators,
    GenerateIndicatorsConfig,
)
from stockdata.utils import get_save_folder_and_file_name, save_parquet_file

ENV_VARS = {
    **dotenv_values(
        "/home/diadochus/Desktop/repos/finance-starter-exp/stockdata/.env.1m"
    )
}


def main(arg_parser):
    # ds, logical date
    airflow_date = arg_parser.execution_date
    # interval
    interval = arg_parser.interval

    # Read the data either from yahoo download or polygon download
    _, yahoo_file = get_save_folder_and_file_name(
        start_date=airflow_date,
        save_base_folder=ENV_VARS["YAHOO_BASE"],
        interval=interval,
    )

    if os.path.isfile(yahoo_file):
        # read the single day data
        df_raw = pd.read_parquet(yahoo_file)
        print("Using yahoo data.")
    else:
        print(f"File doesn't exist for the date in yahoo or polygon: {airflow_date}")
        return 0

    if len(df_raw) == 0:
        print(f"File doesn't have any data: {yahoo_file}")
        return 0

    # generate indicators tic by tic
    df_list = []
    not_processed_counter = 0
    for tic in df_raw.ticker.unique().tolist():
        if not isinstance(tic, str):
            continue
        dft = df_raw.loc[df_raw.ticker == tic]
        # generate indicators
        config = GenerateIndicatorsConfig(
            ticker=tic,
            date=airflow_date,
            close_column="adj_close",
        )
        gi = GenerateIndicators(config=config, df=dft)
        dfi = gi.generate_indicators()
        if len(dfi) == 0:
            not_processed_counter += 1
        df_list.append(dfi)
    print(f"Num of tickers that are not processed: {not_processed_counter}")
    df = pd.concat(df_list, ignore_index=True)
    print("Saving indicator data.")
    save_parquet_file(df, airflow_date, ENV_VARS["INDICATOR_BASE"], arg_parser.interval)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution_date", type=str, help="Execution date received from Airflow"
    )
    parser.add_argument("--interval", type=str, help="Data interval to download 1m 1d")
    main(parser.parse_args())
