import datetime
import pandas as pd
import numpy as np

def timestamp_to_str(ts):
    """تبدیل timestamp به رشته تاریخ-زمان خوانا"""
    return datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def normalize_series(series):
    """نرمال‌سازی یک سری عددی بین 0 و 1"""
    if series.max() == series.min():
        return np.zeros_like(series)
    return (series - series.min()) / (series.max() - series.min())

def load_csv_data(path):
    """بارگذاری داده از فایل CSV"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
