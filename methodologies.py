import typing as tp

import numpy as np
import pandas as pd


def standard_approach(X: np.ndarray,
                      y_true: np.ndarray,
                      y: np.ndarray,
                      model: 'sklearn.base.BaseEstimator',
                      test_size = 0.5,
                      random_state = None
                     ) -> pd.DataFrame:
    """Fits model estimating Pr[response|X; treatment]
    
    Fits model on treatment group and scores both treatment
    and control groups. Returns results for hould out sample.
    
    :param X: array of predictors
    :param y_true: vector of true probabilities
    :param y: observed responses
    :param model: sklearn estimator
    :param test_size: proportion of data used as hold out
    :random_state: random seed
    
    :returns: Dataframe of hold out sample results
    """
    # get remove treatment from predictors
    treatment_mask = X[:,-1].astype(bool)
    X = X[:, :-1]
    
    # split treatment and control
    X_t = X[treatment_mask]
    y_t_true = y_true[treatment_mask]
    y_t = y[treatment_mask]
    y_c_true = y_true[~treatment_mask]
    y_c = y[~treatment_mask]
    
    # fit and score on treatment group
    n_train = int(X_t.shape[0] * test_size)
    n_test = X_t.shape[0] - n_train
    
    train_mask = np.hstack((np.ones(n_train), np.zeros(n_test))).astype(bool)
    np.random.shuffle(train_mask)
    
    X_t_train = X_t[train_mask]
    X_t_test = X_t[~train_mask]
    y_t_train = y_t[train_mask]
    y_t_test = y_t[~train_mask]
    y_t_true = y_t_true[~train_mask]
    
    model.fit(X_t_train, y_t_train)
    y_t_pred = model.predict_proba(X_t_test)[:,1]
    
    # score control group
    y_c_pred = model.predict_proba(X[~treatment_mask])[:,1]
    y_c_true = y_true[~treatment_mask]
    y_c = y[~treatment_mask]
    
    return pd.DataFrame({
        'y': np.hstack((y_t_test, y_c)),
        'y_true': np.hstack((y_t_true, y_c_true)),
        'y_pred': np.hstack((y_t_pred, y_c_pred)),
        'treatment': np.hstack((np.ones(len(y_t_pred)), np.zeros(len(y_c_pred))))
    })
    
    
def true_lift_approach(X: np.ndarray,
                       y_true: np.ndarray,
                       y: np.ndarray,
                       model: 'sklearn.base.BaseEstimator',
                       test_size = 0.5,
                       cross = True,
                       random_state = None
                       ) -> pd.DataFrame:
    """Estimates uplift using the TrueLift approach
    
    Fits model estimating uplift defined as follows:
    Pr[response| X; treatment] - Pr[response| X; no treatment]
    
    :param X: array of predictors
    :param y_true: vector of true probabilities
    :param y: observed responses
    :param model: sklearn estimator
    :param test_size: proportion of data used as hold out
    :param cross: whether to create cross features X*treatment
    :param random_state: random seed
    
    :returns: Dataframe of hold out sample results
    """
    n_train = int(X.shape[0] * test_size)
    n_test = X.shape[0] - n_train
    
    train_mask = np.hstack((np.ones(n_train), np.zeros(n_test))).astype(bool)
    np.random.shuffle(train_mask)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[~train_mask]
    X_test_treat = X_test.copy()
    X_test_treat[:,-1] = 1
    X_test_non_treat = X_test.copy()
    X_test_non_treat[:,-1] = 0
    
    if cross:
        X_train = _add_cross(X_train)
        X_test_non_treat = _add_cross(X_test_non_treat)
        X_test_treat = _add_cross(X_test_treat)
        
    model.fit(X_train, y_train)
    
    pred_non_treat = model.predict_proba(X_test_non_treat)[:,1]
    pred_treat = model.predict_proba(X_test_treat)[:,1]
    
    uplift = pred_treat - pred_non_treat
    
    return pd.DataFrame({
        'y': y[~train_mask],
        'y_true': y_true[~train_mask],
        'y_pred': uplift,
        'treatment': X_test[:,-1]
    })
    
    
def _add_cross(X: np.ndarray) -> np.ndarray:
    """docs
    """
    X = X.copy()
    X_cross = X[:,:-1].copy()

    X_cross[~X[:,-1].astype(bool)] = np.zeros(X_cross.shape[1])
    
    return np.hstack((X, X_cross))


def multi_class_approach(X: np.ndarray,
                         y_true: np.ndarray,
                         y: np.ndarray,
                         model: 'sklearn.base.BaseEstimator',
                         test_size = 0.5,
                         random_state = None
                         ) -> pd.DataFrame:
    """Estimates uplift using multiclass classifier
    
    Derives uplift from 4-class classifier predicting
    all combiantions of treatment/contron and response/
    non response
    
    :param X: array of predictors
    :param y_true: vector of true probabilities
    :param y: observed responses
    :param model: sklearn estimator
    :param test_size: proportion of data used as hold out
    :param random_state: random seed
    
    :returns: Dataframe of hold out sample results
    """
    X = X.copy()
    y = y.copy()
    
    y_multi = y + 2*X[:,-1]
    
    n_train = int(X.shape[0] * test_size)
    n_test = X.shape[0] - n_train
    
    train_mask = np.hstack((np.ones(n_train), np.zeros(n_test))).astype(bool)
    np.random.shuffle(train_mask)
    
    X_train = X[train_mask][:,:-1]
    y_train = y_multi[train_mask]
    X_test = X[~train_mask][:,:-1]
    
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    
    Pr_T = X[:,-1].mean()
    Pr_C = 1 - Pr_T
    
    TR_T = y_pred[:,3] / Pr_T
    CN_C = y_pred[:,0] / Pr_C
    TN_T = y_pred[:,2] / Pr_T
    CR_C = y_pred[:,1] / Pr_C
    
    uplift = (TR_T + CN_C - TN_T - CR_C) / 2
    
    return pd.DataFrame({
        'y': y[~train_mask],
        'y_true': y_true[~train_mask],
        'y_pred': uplift,
        'treatment': X[~train_mask][:,-1]
    })
