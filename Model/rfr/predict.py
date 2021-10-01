import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

SEED = 42
FOLDS = 5
LAG = 5
COL = ['open', 'high', 'low', 'close']

def create_logret(df, cols):
    for col in cols:
        logret_col = col + '_logret'
        df[logret_col] = np.log(df[col]/df[col].shift(1))

def create_lags(df, cols):
    for col in cols:
        data = df.loc[:, col]
        logret_col = col + '_logret'
        for lag in range(1, LAG+1):
            lag_col = f'{logret_col}_lag{lag}'
            df[lag_col] = df[logret_col].shift(lag)

def mape(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)
    return output_errors

def rfr_predict(file_name):
    df = pd.read_csv('./Pool/' + file_name)
    create_logret(df, COL)
    create_lags(df, COL)
    new_cols = df.columns[-24:]
    df['target'] = df['close'].shift(-1) / df['close']
    df.dropna(inplace=True)
    X_train, X_test, t_train, t_test = train_test_split(df.loc[:, new_cols], df.loc[:, 'target'], shuffle=False)

    rfr = RandomForestRegressor(random_state=SEED)
    rfr.fit(X_train, t_train)
    predict = rfr.predict(X_test)
    return predict, t_test

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)
# 銘柄の名前を入れるリスト
list_name = []
# 予測対象日結果を入れるリスト
list_predict = []
list_target = []
for file_name in file_names:
    predict = rfr_predict(file_name)[0]
    target = rfr_predict(file_name)[1].values
    # 銘柄の名前の追加
    list_name.append(file_name.split('_')[2][:-4])
    # 予測対象日結果の追加
    list_predict.append(predict[-1])
    list_target.append(target[-1])

df = pd.DataFrame(list_predict, columns = ["result"] , index = list_name)
df.to_csv('predict_2_rfr_0927.csv')
df2 = pd.DataFrame(list_target, columns = ["target"] , index = list_name)
df_merge = pd.concat([df, df2['target']], axis = 1)
df_merge.to_csv('predict_target_2_rfr_0927.csv')

print(mape(np.array(list_target), np.array(list_predict))*100)