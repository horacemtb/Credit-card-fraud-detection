#!/usr/bin/env python

import json
import argparse

from kafka import KafkaConsumer

import os
import sys
import pickle
import glob

import numpy as np
import pandas as pd

import datetime
import time

import yaml

import random


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-g", "--group_id", required=True, help="kafka consumer group_id"
    )
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
        "-t", "--topic", default="predictions", help="kafka topic to consume predictions"
    )

    args = argparser.parse_args()

    consumer = KafkaConsumer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="YandexCA.crt",
        group_id=args.group_id,
        value_deserializer=json.loads,
    )

    consumer.subscribe(topics=[args.topic])

    print_new_messages(consumer)


def print_new_messages(consumer):
    count = 0
    print("Waiting for a new messages. Press Ctrl+C to stop")
    try:
        for msg in consumer:
           
            print(f"{msg.topic}:{msg.partition}:{msg.offset}: key={msg.key} value={msg.value}")
            count += 1
    except KeyboardInterrupt:
        pass
    print(f"Total {count} messages received")

if __name__ == "__main__":
    main()
