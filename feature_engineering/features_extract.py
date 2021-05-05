import os
from math import ceil

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def extract_cat_dept_id(value, type='cat'):
    split_str = value.split('_')
    if type == 'cat':
        return split_str[0]
    elif type == 'dept':
        return '_'.join(split_str[:2])
    else:
        raise ValueError()


def add_validation(data, end_train):
    index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
    df = pd.melt(
        data, id_vars=index_columns, 
        var_name='d', value_name='sales')

    test_df = pd.DataFrame()
    temp_df = df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['sales'] = np.nan
    for i in range(1,29):
        temp_df['d'] = 'd_'+ str(end_train + i)
        test_df = pd.concat([test_df, temp_df])

    df = pd.concat([df, test_df])
    df.reset_index(drop=True, inplace=True)

    for col in index_columns:
        df[col] = df[col].astype('category')
    return df


def del_unlist_product_sales(data, price_data, calendar_data):
    release_df = price_data.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id', 'item_id', 'release']
    release_df['store_id'] = release_df['store_id'].astype('category')
    release_df['item_id'] = release_df['item_id'].astype('category')

    df = data.merge(release_df, on=['store_id','item_id'])
    df = df.merge(calendar_data[['wm_yr_wk','d']], on=['d'])

    df = df[df['wm_yr_wk'] >= df['release']]
    df = df.reset_index(drop=True)

    df['release'] = df['release'] - df['release'].min()
    df['release'] = df['release'].astype(np.int16)
    return df


def extract_prices_features(data, prices_data, calendar_data):
    # basic stats
    prices_data['price_max'] = prices_data.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices_data['price_min'] = prices_data.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices_data['price_std'] = prices_data.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
    prices_data['price_mean'] = prices_data.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    prices_data['price_norm'] = prices_data['sell_price'] / prices_data['price_max']

    # 同時間點同部門產品價格變異 / 同時間點同部門產品數量 / 同時間點同部門產品價格相同的數量(替代品)
    prices_data['cross_price_std'] = prices_data.groupby(['store_id', 'wm_yr_wk', 'cat_id'])['sell_price'].transform('std')
    prices_data['cross_item_nunique'] = prices_data.groupby(['store_id', 'wm_yr_wk', 'cat_id'])['item_id'].transform('nunique').astype('int16')
    prices_data['cross_same_price_item_nunique'] = prices_data.groupby(['store_id', 'wm_yr_wk', 'cat_id', 'sell_price'])['item_id'].transform('nunique').astype('int16')

    # merge calendar_df in order to aggregate by datetime
    calendar_ymw = calendar_data[['wm_yr_wk', 'month', 'year']]
    calendar_ymw = calendar_ymw.drop_duplicates(subset=['wm_yr_wk'])
    prices_data = prices_data.merge(calendar_ymw[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    del calendar_ymw

    # relative price
    prices_data['price_momentum_w'] = prices_data['sell_price'] / prices_data.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices_data['price_momentum_m'] = prices_data['sell_price'] / prices_data.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
    prices_data['price_momentum_y'] = prices_data['sell_price'] / prices_data.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
    prices_data['cross_rel_price'] = prices_data['sell_price'] / prices_data.groupby(['store_id', 'cat_id'])['sell_price'].transform('mean')

    del prices_data['month'], prices_data['year'], prices_data['cat_id'], prices_data['dept_id']

    # Merge Prices
    main_price_df = data.merge(prices_data, on=['store_id','item_id','wm_yr_wk'], how='left')
    keep_columns = [col for col in list(main_price_df) if col not in list(data)]
    return main_price_df[MAIN_INDEX + keep_columns]


def extract_datetime_features(data, calendar_data):
    feature_cols = [
        'date', 'd',
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI']
    nan_cols = [
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2']

    # convert the data type and fill missing value
    main_cal_df = data[MAIN_INDEX].merge(calendar_data[feature_cols], on='d', how='left')
    main_cal_df['date'] = pd.to_datetime(main_cal_df['date'])
    for column in feature_cols[2:]:
        if column in nan_cols:
            main_cal_df[column].fillna('unknown', inplace = True)
        main_cal_df[column] = main_cal_df[column].astype('category')

    # datetime features
    main_cal_df['tm_d'] = main_cal_df['date'].dt.day.astype(np.int8)
    main_cal_df['tm_w'] = main_cal_df['date'].dt.week.astype(np.int8)
    main_cal_df['tm_m'] = main_cal_df['date'].dt.month.astype(np.int8)
    main_cal_df['tm_y'] = main_cal_df['date'].dt.year
    main_cal_df['tm_y'] = (main_cal_df['tm_y'] - main_cal_df['tm_y'].min()).astype(np.int8)
    main_cal_df['tm_dw'] = main_cal_df['date'].dt.dayofweek.astype(np.int8)
    main_cal_df['tm_w_end'] = (main_cal_df['tm_dw'] >= 5).astype(np.int8)

    del main_cal_df['date']
    return main_cal_df


def extract_sales_features(data):
    groups =  [
        ['state_id'],
        ['store_id'],
        ['cat_id'],
        ['dept_id'],
        ['item_id'],
        ['state_id', 'cat_id'],
        ['store_id', 'cat_id']
    ]
    main_sales_df = data.copy()

    # lag sales features
    main_sales_df['sales_lag_month'] = main_sales_df.groupby(['id'])['sales'].transform(lambda x: x.shift(28))
    for group_column in groups:
        column = '_'.join(group_column)
        main_sales_df[column + '_lag_mean'] = main_sales_df.groupby(group_column)['sales'].transform('mean')
        main_sales_df[column + '_lag_mean'].astype(np.float16, inplace=True)
        main_sales_df[column + '_lag_std'] = main_sales_df.groupby(group_column)['sales'].transform('std')
        main_sales_df[column + '_lag_std'].astype(np.float16, inplace=True)

    # convert data type
    main_sales_df['d'] = main_sales_df['d'].apply(lambda x: x[2:]).astype(np.int16)
    del main_sales_df['wm_yr_wk']
    return main_sales_df


def transform_category_features(data):
    category_cols = [
        'item_id', 'dept_id', 'cat_id',
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2']

    for column in category_cols:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[feature]).astype('int16')
    return data
