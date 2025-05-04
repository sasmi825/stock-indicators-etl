# stock-indicators-etl
Pulls 1m stock data from Yahoo Finance and generates stock momentum indicators. Used Airflow for orchestration. Airflow DAG can be found in `airflow_dag.py` file. There are two tasks in this Airflow DAG, Downloader and Indicators. Indicators task depends on Downloader task. This DAG is scheduled to run every weekday at 9:30 AM to download previous day's stock data.

## Library Structure
`data_download_yahoo.py`: A script to download 1-minute (1m) or 1-day (1d) historical stock data. Takes execution_date and interval arguments.
`data_indicators.py`: A script to convert raw stock data to stock momentum indicators. Used in Airflow DAG. Takes execution_date and interval arguments.
`stockdata/indicators.py`: A script that uses TA-Lib to convert stock data to indicators.

Following momentum stock indicators can be generated:
```
momentum_features = [
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
```
