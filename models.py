import lightgbm as lgb


def lgb_model(params, mode):

    if mode == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    return model
