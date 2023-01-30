import json
from loguru import logger
from fastapi import FastAPI
import pickle

import pandas as pd

from app.class_Transaction import Transaction
from app.preprocess_new_transaction import preprocess_transaction
from starlette_exporter import PrometheusMiddleware, handle_metrics

logger.debug('Running app')

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

model_path = './app/models/RF.pickle'
DIR_PROC = './app/proc-data/'
customer_columns = ['CUSTOMER_ID', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                    'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                    'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
    
terminal_columns = ['TERMINAL_ID', 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
                    'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
                    'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']

input_features = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']

logger.debug('Loading model')
predictor = pickle.load(open(model_path, 'rb'))

logger.debug('Model loaded')


@app.post('/predict/')
async def predict_tr(tr: Transaction):

    try:

        logger.debug('Processing transaction')
        proc_tr = preprocess_transaction(tr.transaction, DIR_PROC, customer_columns, terminal_columns)
        
        logger.debug('Making inference')
        pred = predictor.predict(proc_tr[input_features])[0]
        proba = predictor.predict_proba(proc_tr[input_features])[0][1]
        
        logger.debug('Inference complete')
        
        proc_tr['prediction'] = pred
        proc_tr['probability'] = proba
        
        res = proc_tr[['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'prediction', 'probability']].iloc[0, :].to_dict()
        
        res['TRANSACTION_ID'] = int(res['TRANSACTION_ID'])
        res['TX_DATETIME'] = str(res['TX_DATETIME'])
        res['CUSTOMER_ID'] = int(res['CUSTOMER_ID'])
        res['TERMINAL_ID'] = int(res['TERMINAL_ID'])
        res['prediction'] = int(res['prediction'])
        
        tr.predictions = json.dumps(res)
    
    except Exception as ex:
        logger.error(f'Error getting predictions in main.py: {ex}')

    return tr.predictions
