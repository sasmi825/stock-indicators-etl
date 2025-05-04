from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


PYTHON_ENV = "/home/diadochus/anaconda3/envs/finexp/bin/python"
BASE_FOLDER = "/home/diadochus/Desktop/repos/stock-indicators-etl"

with DAG(
    "Stock-Indicators",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "email": ["sasmipolu@gmail.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 1,  # will be over-written by task lvl setting.
        "retry_delay": timedelta(minutes=5),
    },
    description="Downloads 1m interval stock data from yahoo and generates stock indicators",
    schedule="30 9 * * 1-5",  # runs every week day at 9:30 AM
    start_date=datetime(2024, 9, 30),
    catchup=True,
    tags=["stocks"],
    max_active_runs=8,
) as dag:
    t1_base = f"{PYTHON_ENV} {BASE_FOLDER}/data_download_yahoo.py"
    t1 = BashOperator(
        task_id="Downloader",
        depends_on_past=False,
        bash_command="{cmd} --execution_date '{{{{ ds }}}}' --interval 1m".format(cmd=t1_base),
        retries=3,
    )

    t2_base = f"{PYTHON_ENV} {BASE_FOLDER}/data_indicators.py"
    t2 = BashOperator(
        task_id="Indicators",
        depends_on_past=False,
        bash_command="{cmd} --execution_date '{{{{ ds }}}}' --interval 1m".format(cmd=t2_base),
        retries=3,
    )

    t1 >> t2
