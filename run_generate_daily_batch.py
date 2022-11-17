from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator

args = {'owner': 'airflow'}

dag = DAG(dag_id = 'generate_daily_batch', default_args = args, schedule_interval = '59 23 * * *', start_date = days_ago(0), tags = ['SSH'])

sshHook = SSHHook(remote_host = '51.250.106.71', username = 'ubuntu', key_file = r'/home/dmitryairflow/sshn/id_rsa')

generate_daily_batch = SSHOperator(ssh_hook = sshHook, task_id = 'task1', command = 'ipython3 /home/ubuntu/Fraud-detection/generate_daily_batch.py', dag=dag)

generate_daily_batch