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

def rfr_model(file_name):
    df = pd.read_csv('./Pool/' + file_name)
    folds = TimeSeriesSplit(n_splits=FOLDS)
    create_logret(df, COL)
    create_lags(df, COL)
    new_cols = df.columns[-24:]
    df['target'] = df['close'].shift(-1) / df['close']
    df['naive_pred'] = df['target'].shift(1)
    df.dropna(inplace=True)
    scores_naive, scores_rfr = [], []
    for i, (train_idx, val_idx) in enumerate(folds.split(df)):
        df_train = df.iloc[train_idx, :]
        df_val = df.iloc[val_idx, :]
        X_train, t_train = df_train.loc[:, new_cols], df_train.loc[:, 'target']
        X_val, t_val = df_val.loc[:, new_cols], df_val.loc[:, 'target']

        naive_val = df_val.loc[:, 'naive_pred']
        score_naive = mape(t_val, naive_val)
        scores_naive.append(score_naive)

        rfr = RandomForestRegressor(random_state=SEED)
        rfr.fit(X_train, t_train)
        y_val = rfr.predict(X_val)
        score_rfr = mape(t_val, y_val)
        scores_rfr.append(score_rfr)
    return scores_naive, scores_rfr

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)
cvs_naive, cvs_rfr = [], []
for file_name in file_names:
    scores_naive = rfr_model(file_name)[0]
    cv_ave_naive = sum(scores_naive)/len(scores_naive)
    cv_std_naive = np.std(scores_naive)
    cvs_naive.append(cv_ave_naive)

    scores_rfr = rfr_model(file_name)[1]
    cv_ave_rfr = sum(scores_rfr)/len(scores_rfr)
    cv_std_rfr = np.std(scores_rfr)
    cvs_rfr.append(cv_ave_rfr)

cvs_ave_naive = sum(cvs_naive)/len(cvs_naive)
cvs_std_naive = np.std(cvs_naive)
cvs_ave_rfr = sum(cvs_rfr)/len(cvs_rfr)
cvs_std_rfr = np.std(cvs_rfr)

print('CV Average :')
print(f' naive : {np.round(cvs_ave_naive*100, 2)} ± {np.round(cvs_std_naive*100, 2)}%')
print(f' rfr : {np.round(cvs_ave_rfr*100, 2)} ± {np.round(cvs_std_rfr*100, 2)}%')