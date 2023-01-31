import datetime
import requests
from generate_new_transaction import get_latest_data, seconds_since_midnight, generate_single_transaction

DIR_RAW = '/home/dmitry-ds/credit-card-fraud-detection/Credit-card-fraud-detection/data/'

latest_data = get_latest_data(DIR_RAW)

N_CUSTOMERS = latest_data['CUSTOMER_ID'].nunique() 
N_TERMINALS = latest_data['TERMINAL_ID'].nunique()
new_id = latest_data.iloc[-1, :]['TRANSACTION_ID']+1
new_day = latest_data.iloc[-1, :]['TX_TIME_DAYS']+1
today = latest_data.iloc[-1, :]['TX_DATETIME'] + datetime.timedelta(days=1)
todays_date = f'{today.year}-{today.month}-{today.day}'

new_sec = seconds_since_midnight()
tr = generate_single_transaction(new_id, new_sec, new_day, todays_date, N_CUSTOMERS, N_TERMINALS, latest_data)

print(tr)

r = requests.post('http://51.250.38.128:80/predict/', json={'transaction': tr})

print(r.json())