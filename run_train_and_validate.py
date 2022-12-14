from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator

args = {'owner': 'airflow'}

dag = DAG(dag_id = 'run_train_and_validate', default_args = args, schedule_interval = '10 0 * * *', start_date = days_ago(0), tags = ['SSH'])

sshHook = SSHHook(remote_host = '158.160.13.60', username = 'ubuntu', key_file = r'/home/dmitryairflow/sshn/id_rsa')

run_train = SSHOperator(ssh_hook = sshHook, task_id = 'task3', command = 'sudo su - ubuntu -c "ipython3 /home/ubuntu/Fraud-detection/ML_pyspark_automatic_validation.py" ', dag=dag)

run_train