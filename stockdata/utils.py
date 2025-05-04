import datetime as dt
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import pytz
from numpy.typing import NDArray

# divide by this to get seconds from the window_start difference
WINDOW_DIVIDER = 10**9  # 1000000000


def get_unix_timestamp(date_time, timezone):
    """Gets the Unix timestamp for a given datetime and timezone."""

    # Create a timezone-aware datetime object
    tz = pytz.timezone(timezone)
    dt_aware = tz.localize(date_time)

    # Convert to UTC and get timestamp
    dt_utc = dt_aware.astimezone(pytz.utc)
    return int(dt_utc.timestamp()) * 10**9


def get_market_open_close_unix(data_date: str):
    """Get market open and close unix timestamps for a given date."""
    est_start_time = dt.datetime.strptime(f"{data_date} 09:30:00", "%Y-%m-%d %H:%M:%S")
    # market start timestamp
    mst = get_unix_timestamp(est_start_time, "US/Eastern")

    est_end_time = dt.datetime.strptime(f"{data_date} 16:30:00", "%Y-%m-%d %H:%M:%S")
    # market end timestamp
    met = get_unix_timestamp(est_end_time, "US/Eastern")

    return mst, met


def list_all_files(mypath: str) -> list[str]:
    """
    Lists all files in mypath

    :param mypath: str, folder path
    :return: list, list of all files in mypath folder
    """
    only_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return only_files


def create_sub_folder(path: str) -> None:
    """
    Creates a folder if it doesn't exist. Can create parent directories.

    :param path: str, folder path to be created
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_save_folder_and_file_name(
    start_date: str,  # "%Y-%m-%d"
    save_base_folder: str,
    interval: str,
    polygon_file=False,
):
    yr, mo, da = start_date.split("-")
    save_folder = f"{save_base_folder}/{interval}/{yr}/{mo}"
    if polygon_file:
        save_file = f"{save_folder}/{start_date}.csv.gz"
    else:
        save_file = f"{save_folder}/{start_date}.parquet.gzip"
    return save_folder, save_file


def save_parquet_file(
    df: pd.DataFrame,
    start_date: str,  # "%Y-%m-%d"
    save_base_folder: str,
    interval: str,
) -> None:
    save_folder, save_file = get_save_folder_and_file_name(
        start_date=start_date, save_base_folder=save_base_folder, interval=interval
    )
    create_sub_folder(save_folder)
    df.to_parquet(save_file, compression="gzip", index=False)


def break_chunks(arr: NDArray, chunk_size: int) -> list:
    """
    Breaks a 2D numpy array into chunks of fixed length along the first dimension.

    Args:
        arr: The 2D numpy array to be chunked.
        chunk_size: The desired size of each chunk along the first dimension.

    Returns:
        A list of numpy arrays, where each element is a chunk of the original array.
    """
    chunks = []
    n_chunks = arr.shape[0] // chunk_size
    for i in range(n_chunks):
        chunks.append(arr[i * chunk_size : (i + 1) * chunk_size])
    # Handle remaining elements if the array size is not divisible by chunk_size
    if arr.shape[0] % chunk_size != 0:
        chunks.append(arr[n_chunks * chunk_size :])
    return chunks


def get_week_year(date: str, date_format: str = "%Y-%m-%d") -> tuple:
    """
    For a given date, returns week of the year and year

    :param date: str, date for which wk of yr and yr is required.
    :param date_format: str, (default="%Y-%m-%d") date format
    :return: tuple(wk_of_yr, yr)
    """
    dd_ = dt.datetime.strptime(date, date_format).isocalendar()
    yr = dd_.year
    wk = dd_.week
    return wk, yr
