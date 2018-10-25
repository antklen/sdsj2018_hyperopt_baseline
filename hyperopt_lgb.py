import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

from models import lgb_model


def hyperopt_lgb(X, y, mode, N, time_limit, max_train_size=None, max_train_rows=None):
    """hyperparameters optimization with hyperopt"""

    print('hyperopt..')

    start_time = time.time()

    # train-test split
    train_size = 0.7
    # restrict size of train set to be not greater than max_train_size
    if max_train_size is not None:
        size_factor = max(1, 0.7*X.memory_usage(deep=True).sum()/max_train_size)
    # restrict number of rows in train set to be not greater than max_train_rows
    if max_train_rows is not None:
        rows_factor = max(1, 0.7*X.shape[0]/max_train_rows)
    train_size = train_size/max(size_factor, rows_factor)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size, random_state=42)
    print('train shape {}, size {}'.format(Xtrain.shape, Xtrain.memory_usage(deep=True).sum()/1024/1024))

    # search space to pass to hyperopt
    fspace = {
        'num_leaves': hp.choice('num_leaves', [5,10,20,30,50,70,100]),
        'subsample': hp.choice('subsample', [0.7,0.8,0.9,1]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1]),
        'min_child_weight': hp.choice('min_child_weight', [5,10,15,20,30,50]),
        'learning_rate': hp.choice('learning_rate', [0.02,0.03,0.05,0.07,0.1,0.2]),
    }

    # objective function to pass to hyperopt
    def objective(params):

        iteration_start = time.time()

        # print(params)
        params.update({'n_estimators': 500, 'random_state': 42, 'n_jobs': -1})

        model = lgb_model(params, mode)
        model.fit(Xtrain, ytrain)

        if mode == 'regression':
            pred = model.predict(Xtest)
            loss = np.sqrt(mean_squared_error(ytest, pred))
        elif mode == 'classification':
            pred = model.predict_proba(Xtest)[:, 1]
            loss = -roc_auc_score(ytest, pred)

        iteration_time = time.time()-iteration_start
        print('iteration time %.1f, loss %.5f' % (iteration_time, loss))

        return {'loss': loss, 'status': STATUS_OK,
                'runtime': iteration_time,
                'params': params}


    # object with history of iterations to pass to hyperopt
    trials = Trials()

    # loop over iterations of hyperopt
    for t in range(N):
        # run hyperopt, n_startup_jobs - number of first iterations with random search
        best = fmin(fn=objective, space=fspace, algo=partial(tpe.suggest, n_startup_jobs=10),
                    max_evals=t+1, trials=trials)
        # check if time limit exceeded, then interrupt search
        elapsed = time.time()-start_time
        if elapsed >= time_limit:
            print('time limit exceeded')
            break

    print('best parameters', trials.best_trial['result']['params'])

    return trials.best_trial['result']['params']
