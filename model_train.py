import os, random

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

SEED = 9568
LABELS = 'sales'
START_TRAIN = 0
END_TRAIN = 1913
HORIZON = 28

TRAIN_DATA_DIR = '~/M5/train_df.pkl'
EVALUATION_DATA_DIR = '~/M5/evaluation_df.pkl'
VALIDATION_Y_DIR = '~/M5/validation_y_df.pkl'
MODEL_DIR = '~/M5/model'

# FEATURES to remove
STORE_IDS = [
    'CA_1', 'CA_2', 'CA_3', 'CA_4',
    'TX_1', 'TX_2', 'TX_3',
    'WI_1', 'WI_2', 'WI_3']
CAT_IDS = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
REMOVE_FEATURES = [
    'id', 'd', LABELS,
    'state_id', 'store_id',
    'cat_id', 'date', 'wm_yr_wk']


def get_train_by_cat_store(store, cat):
    df = pd.read_pickle(TRAIN_DATA_DIR)
    df = df[(df['cat_id']==cat) & (df['store_id']==store)]

    # features
    features = [col for col in list(df) if col not in REMOVE_FEATURES]
    df = df[['id', 'd', LABELS] + features]
    return df.reset_index(), features


def get_evaluation_by_cat_store(store, cat):
    df = pd.read_pickle(EVALUATION_DATA_DIR)
    df = df[(df['cat_id']==cat) & (df['store_id']==store)]

    # features
    features = [col for col in list(df) if col not in REMOVE_FEATURES]
    df = df[['id', 'd', LABELS] + features]
    return df.reset_index(), features


def get_validation_y_by_cat_store(store, cat):
    df = pd.read_pickle(VALIDATION_Y_DIR)
    df = df[(df['cat_id']==cat) & (df['store_id']==store)]
    return df['sales'].values


def train(lgb_params)
    valid_y_list = []
    valid_y_hat_list = []
    tmp_result_df = []

    for cat in CAT_IDS:
        for store in STORE_IDS:
            total_df, features = get_train_by_cat_store(store, cat)

            train_mask = total_df['d'] <= END_TRAIN
            lgb_valid_mask = grid_df['d'] > (END_TRAIN - HORIZON)

            train = total_df[train_mask].dropna()
            valid = total_df[lgb_valid_mask].dropna()

            train_data = lgb.Dataset(
                train[features], label=train[LABELS])
            valid_data = lgb.Dataset(
                valid[features], label=valid[LABELS])

            # Train Models
            print('%s-%s' % (cat, store))
            estimator = lgb.train(
                lgb_params, train_data,
                valid_sets=[valid_data],
                verbose_eval=False)

            # Validation
            validation_mask = total_df['d'] > END_TRAIN
            validation_X = total_df[validation_mask][features]
            validation_y = get_validation_y_by_cat_store(store, cat)
            validation_y_hat = estimator.predict(test_X)

            temp_validation_df = pd.DataFrame({
                'id': total_df[validation_mask]['id'],
                'd': ['F' + str(d - END_TRAIN) for d in total_df[validation_mask]['d']],
                'prediction': validation_y_hat})

            tmp_result_df.append(temp_validation_df)
            valid_y_hat_list.extend(validation_y_hat.tolist())
            valid_y_list.extend(validation_y.tolist())

            # Evaluation
            evaluation_X, _ = get_evaluation_by_cat_store(store, cat)
            evaluation_y_hat = estimator.predict(evaluation_X[features])

            temp_eval_df = pd.DataFrame({
                'id': evaluation_X['id'],
                'd': ['F' + str(d - END_TRAIN + 28) for d in evaluation_X['d']],
                'prediction': evaluation_y_hat})

            tmp_result_df.append(temp_eval_df)

            model_name = MODEL_DIR + store + '_' + cat + '_.bin'
            pickle.dump(estimator, open(model_name, 'wb'))

    pred_df = pd.concat(tmp_result_df).reset_index(drop=True)
    return valid_y, valid_y_hat, pred_df


if __name__ == '__main__':
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'learning_rate': 0.05,
        'feature_fraction': 0.5,
        'max_bin': 255,
        'num_iterations': 1000,
        'verbose': -1,
        'seed': SEED}
    
    valid_y, valid_y_hat, pred_df = train(lgb_params)
    mean_squared_error(valid_y, valid_y_hat, squared=False)
    
    pred_df = pred_df.pivot(index='id', columns='d', values='prediction').reset_index()

    cols = ['id'] + ['F' + str(i) for i in range(1, 29)]
    submission = pd.read_csv('~/M5/sample_submission.csv')[['id']]
    submission = submission.merge(result_df, on='id', how='left')[cols]
    submission.to_csv('submission.csv', index=False)
