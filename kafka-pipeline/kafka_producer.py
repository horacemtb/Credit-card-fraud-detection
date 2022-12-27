#!/usr/bin/env python

import json
from typing import Dict, NamedTuple
import argparse
from collections import namedtuple
import logging

import os
import sys
import pickle
import glob

import numpy as np
import pandas as pd

import datetime
import time

import random

from tqdm import tqdm
tqdm.pandas()

from generate_new_transaction import get_latest_data, seconds_since_midnight, generate_single_transaction

import kafka


#DIR_RAW = "D:/DS/MLOps/Credit-card-fraud-detection/simulated-data-raw/"
DIR_RAW = "/home/ubuntu/Fraud-detection/simulated-data-raw/"

latest_data = get_latest_data(DIR_RAW)

N_CUSTOMERS = latest_data['CUSTOMER_ID'].nunique() 
N_TERMINALS = latest_data['TERMINAL_ID'].nunique()
new_id = latest_data.iloc[-1, :]['TRANSACTION_ID']+1
new_day = latest_data.iloc[-1, :]['TX_TIME_DAYS']+1
today = latest_data.iloc[-1, :]['TX_DATETIME'] + datetime.timedelta(days=1)
todays_date = f'{today.year}-{today.month}-{today.day}'


class RecordMetadata(NamedTuple):
    topic: str
    partition: int
    offset: int


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-b",
        "--bootstrap_server",
        default="rc1b-tdjco20jo3dnkt2m.mdb.yandexcloud.net:9091",
        help="kafka server address:port",
    )
    argparser.add_argument(
        "-u", "--user", required=True, default="user", help="kafka user"
    )
    argparser.add_argument(
        "-p", "--password", required=True, default="password", help="kafka user password"
    )
    argparser.add_argument(
        "-t", "--topic", default="raw-data", help="kafka topic to produce raw data"
    )
    argparser.add_argument(
        "-n",
        default=10,
        type=int,
        help="number of messages to send",
    )

    args = argparser.parse_args()

    producer = kafka.KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="YandexCA.crt",
        value_serializer=serialize,
    )

    global DIR_RAW
    global latest_data
    global N_CUSTOMERS
    global N_TERMINALS
    global new_id
    global new_day
    global today
    global todays_date

    try:
        for i in range(args.n):
            new_sec = seconds_since_midnight()
            record_md = send_message(producer, args.topic, new_id, new_sec, new_day, todays_date, N_CUSTOMERS, N_TERMINALS, latest_data)
            print(
                f"Msg sent. Topic: {record_md.topic}, partition:{record_md.partition}, offset:{record_md.offset}"
            )
            new_id+=1
    except kafka.errors.KafkaError as err:
        logging.exception(err)
    producer.flush()
    producer.close()


def send_message(producer: kafka.KafkaProducer, topic: str, new_id: int, new_sec: int, new_da: int, todays_date: pd.Timestamp, N_CUSTOMERS: int, N_TERMINALS: int, latest_data: pd.Timestamp) -> RecordMetadata:
    
    tr = generate_single_transaction(new_id, new_sec, new_day, todays_date, N_CUSTOMERS, N_TERMINALS, latest_data)

    future = producer.send(
        topic=topic,
        key=str(tr['TRANSACTION_ID']).encode("ascii"),
        value=str(tr),
    )

    # Block for 'synchronous' sends
    record_metadata = future.get(timeout=1)
    return RecordMetadata(
        topic=record_metadata.topic,
        partition=record_metadata.partition,
        offset=record_metadata.offset,
    )


def serialize(msg: Dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


if __name__ == "__main__":
    main()
