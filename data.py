import typing as tp

import numpy as np


def generate_logistic_data(test_coefs: tp.List[float],
                           control_coefs: tp.List[float],
                           n_test: int,
                           n_control: int,
                           mu: np.ndarray = None,
                           cov: np.ndarray = None
                          ) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Creates dataset based on logistic models
    
    Generates dataset of multinomial normal distributed features
    with logistic response based on two different logit models for
    test and control groups. Returns X - a matrix of features with
    treatment indicator in last column, y_true - vector of ground truth
    probabilities and y - observed values generated from Bernouli
    distribution using y_true.
    
    :param test_coefs: coefficients of test model (intercept first)
    :param control_coefs: coefficients of control model (intercept first)
    :param n_test: size of test group
    :param n_control: size of control group
    :param mu: mean of the multinomial distribution (vector of 0s by default)
    :param cov: covariance matrix of the distribution (identity by default)
    
    :returns: X, y_true, y
    """
    
    if mu is None:
        mu = np.zeros(len(test_coefs)-1)
    if cov is None:
        cov = np.eye(len(test_coefs)-1)
        
    assert len(test_coefs) == len(control_coefs), \
            'Test and control coefs must be of same lenght'
    assert len(mu) == len(test_coefs)-1, \
            'mu must be of same lenght as coefs'
    assert cov.shape == (len(test_coefs)-1, len(test_coefs)-1), \
            'Covariance matrix must be of shape (len(mu), len(mu))'
                        
    X = np.random.multivariate_normal(mu, cov, n_test + n_control)
    # add column vector of 1s for intercept
    X_ = np.hstack(
        (np.ones(n_test + n_control).reshape(-1, 1), X)
    )
    # generate the logistic response
    y_test = 1 / (1 + np.exp(-X_[:n_test].dot(np.array(test_coefs))))
    y_control = 1 / (1 + np.exp(-X_[n_test:].dot(np.array(control_coefs))))
    # generate the treatment indicator
    treatment = np.hstack(
        (np.ones(n_test), np.zeros(n_control))
    )
    
    X = np.hstack((X, treatment.reshape(-1, 1)))
    y_true = np.hstack((y_test, y_control))
    y = np.random.binomial(n=1, p=y_true)
    
    return X, y_true, y
