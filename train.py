import argparse
import os
import pickle
import time
import pandas as pd
import gc

import warnings
warnings.filterwarnings("ignore")

from preprocess import preprocess
from feature_selection import lgb_importance_fs
from hyperopt_lgb import hyperopt_lgb
from models import lgb_model


# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
BIG_DATASET_SIZE = 300 * 1024 * 1024

# hyperopt settings
HYPEROPT_NUM_ITERATIONS = 50
HYPEROPT_MAX_TRAIN_SIZE = 10 * 1024 * 1024
HYPEROPT_MAX_TRAIN_ROWS = 10000

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()


    # read small amount of data to parse dtypes and find datetime columns
    df0 = pd.read_csv(args.train_csv, nrows=5000)
    dtypes = df0.dtypes.map(lambda x: 'float32' if x=='float64' else x).to_dict()
    datetime_cols = df0.columns[df0.columns.str.contains('datetime')].tolist()
    # read full data with float32 instead of float64 and parsing datetime columns
    df = pd.read_csv(args.train_csv, dtype=dtypes, parse_dates=datetime_cols)
    # df = pd.read_csv(args.train_csv)

    y = df.target
    df.drop('target', axis=1, inplace=True)
    is_big = df.memory_usage(deep=True).sum() > BIG_DATASET_SIZE

    print('Dataset read, shape {}'.format(df.shape))
    print('memory_usage {}'.format(df.memory_usage(deep=True).sum()/1024/1024))
    print('time elapsed: {}'.format(time.time()-start_time))

    # dict with data necessary to make predictions
    model_config = {}
    model_config['is_big'] = is_big
    model_config['mode'] = args.mode
    model_config['dtypes'] = dtypes
    model_config['datetime_cols'] = datetime_cols

    # preprocessing
    df, model_config = preprocess(df, model_config, type='train')
    print('number of features {}'.format(len(model_config['used_columns'])))
    print('time elapsed: {}'.format(time.time()-start_time))

    gc.collect()

    # feature selection
    if is_big or len(model_config['used_columns']) > 500:
        df, used_columns = lgb_importance_fs(df, y, args.mode, BIG_DATASET_SIZE)
        model_config['used_columns'] = used_columns
        print('time elapsed: {}'.format(time.time()-start_time))

    # final data shape
    print('final df shape {}'.format(df.shape))

    # hyperopt
    elapsed = time.time()-start_time
    params = hyperopt_lgb(df, y, mode=args.mode, N=HYPEROPT_NUM_ITERATIONS,
                          time_limit=int((TIME_LIMIT-elapsed)*0.7),
                          max_train_size=HYPEROPT_MAX_TRAIN_SIZE,
                          max_train_rows=HYPEROPT_MAX_TRAIN_ROWS)

    # training
    model = lgb_model(params, args.mode)
    model.fit(df, y)
    model_config['model'] = model

    # save config to file
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
