import os
import pandas as pd

DIR_PATH = "./Data/data/BinanceData_d/0924/"
file_names = os.listdir(DIR_PATH)

MIN_RECORDS = 600
FREQ = 'd'  # ['d', '1h', 'minute']
COL = ['open', 'high', 'low', 'close']
FOLDS = 5
SEED = 42

def execute(file_name):
    file_path = DIR_PATH + file_name
    df_orig = pd.read_csv(file_path, header = 1)

    if df_orig.shape[0] < MIN_RECORDS:
        return 1
    
    df_orig = df_orig.sort_values('date').reset_index(drop=True)
    df_ohlc = df_orig.loc[:, ['date'] + COL]

    oc_ratio = (df_ohlc['close']/df_ohlc['open'])
    oc_ratio_bin = oc_ratio.apply(lambda x: 1 if x>=1 else 0)
    df_ohlc['target'] = oc_ratio_bin.shift(-1)
    df_ohlc['naive_pred'] = oc_ratio.apply(lambda x: 0 if x>=1 else 1).shift(1)
    df_ohlc.dropna(inplace=True) 
    df_ohlc.to_csv('./Pool/ohlc_' + file_name) 
    #return df_ohlc
    return 0

file_names = os.listdir(DIR_PATH)
# ステーブルコインペアの削除
file_names.remove('Binance_USDCUSDT_d.csv')
file_names.remove('Binance_TUSDUSDT_d.csv')
file_names.remove('Binance_PAXUSDT_d.csv')

for file_name in file_names:
    execute(file_name)