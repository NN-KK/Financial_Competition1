import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
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

def rfc_model(file_name):
    df = pd.read_csv('./Pool/' + file_name)
    folds = TimeSeriesSplit(n_splits=FOLDS)
    create_logret(df, COL)
    create_lags(df, COL)
    new_cols = df.columns[-24:]
    df.dropna(inplace=True)
    scores_naive, scores_rfc = [], []
    for i, (train_idx, val_idx) in enumerate(folds.split(df)):
        df_train = df.iloc[train_idx, :]
        df_val = df.iloc[val_idx, :]
        X_train, t_train = df_train.loc[:, new_cols], df_train.loc[:, 'target']
        X_val, t_val = df_val.loc[:, new_cols], df_val.loc[:, 'target']

        naive_val = df_val.loc[:, 'naive_pred']
        auc_naive = roc_auc_score(t_val, naive_val)
        scores_naive.append(auc_naive)

        rfc = RandomForestClassifier(random_state=SEED)
        rfc.fit(X_train, t_train)
        y_val = rfc.predict_proba(X_val)[:, 1]
        auc_rfc = roc_auc_score(t_val, y_val)
        scores_rfc.append(auc_rfc)

    return scores_naive, scores_rfc
DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)

cvs_naive, cvs_rfc = [], []
for file_name in file_names:
    scores_naive = rfc_model(file_name)[0]
    cv_ave_naive = sum(scores_naive)/len(scores_naive)
    cv_std_naive = np.std(scores_naive)
    cvs_naive.append(cv_ave_naive)

    scores_rfc = rfc_model(file_name)[1]
    cv_ave_rfc = sum(scores_rfc)/len(scores_rfc)
    cv_std_rfc = np.std(scores_rfc)
    cvs_rfc.append(cv_ave_rfc)

cvs_ave_naive = sum(cvs_naive)/len(cvs_naive)
cvs_std_naive = np.std(cvs_naive)
cvs_ave_rfc = sum(cvs_rfc)/len(cvs_rfc)
cvs_std_rfc = np.std(cvs_rfc)

print('CV Average :')
print(f' naive : {np.round(cvs_ave_naive, 2)} ± {np.round(cvs_std_naive, 2)}')
print(f' rfc      : {np.round(cvs_ave_rfc, 2)} ± {np.round(cvs_std_rfc, 2)}')