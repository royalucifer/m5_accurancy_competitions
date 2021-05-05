import os

import numpy as np
import pandas as pd

import features_extract as fe

END_TRAIN = 1913
TRAIN_DATA_DIR = '~/M5/train_df.pkl'


def main(sales_df, prices_df, calendar_df):
    # basic process
    main_df = fe.add_validation(sales_df, END_TRAIN)
    main_df = fe.del_unlist_product_sales(main_df, prices_df, calendar_df)
    del sales_df

    # extract feature
    main_prices_df = fe.extract_prices_features(main_df, prices_df, calendar_df)
    main_calendar_df = fe.extract_datetime_features(main_df, calendar_df)
    main_sales_df = fe.extract_sales_features(main_df)
    del main_df, calendar_df, prices_df

    # combine all features
    train_df = pd.concat(
        [main_sales_df, main_prices_df.iloc[:, 2:], main_calendar_df.iloc[:, 2:]],
        axis=1)
    del main_sales_df, main_prices_df, main_calendar_df
    
    # transform category features
    train_df = fe.transform_category_features(train_df)
    train_df.to_pickle(TRAIN_DATA_DIR)


if __name__ == '__main__':
    sales_df = pd.read_csv('~/M5/sales_train_validation.csv')
    prices_df = pd.read_csv('~/M5/sell_prices.csv')
    calendar_df = pd.read_csv('~/M5/calendar.csv')

    prices_df['cat_id'] = prices_df['item_id'].apply(fe.extract_cat_dept_id, type='cat')
    prices_df['dept_id'] = prices_df['item_id'].apply(fe.extract_cat_dept_id, type='dept')

    main(sales_df, prices_df, calendar_df)
