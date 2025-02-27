import pandas as pd
from utils.clickhouse_data import clickhouse_largecap, clickhouse_midcap, clickhouse_smallcap

def load_all_data(start_date, end_date, size=10000):
    """Loads market data from ClickHouse"""
    large_cap = clickhouse_largecap(start_date, end_date, size)
    mid_cap = clickhouse_midcap(start_date, end_date, size)
    small_cap = clickhouse_smallcap(start_date, end_date, size)

    return pd.concat([large_cap, mid_cap, small_cap]) 

def load_large_cap(start_date, end_date, size):
    """Loads market data from ClickHouse"""
    return clickhouse_largecap(start_date, end_date, size)

def load_mid_cap(start_date, end_date, size):
    """Loads market data from ClickHouse"""
    return clickhouse_midcap(start_date, end_date, size)

def load_small_cap(start_date, end_date, size):
    """Loads market data from ClickHouse"""
    return clickhouse_smallcap(start_date, end_date, size)