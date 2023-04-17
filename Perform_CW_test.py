# -*- coding: utf-8 -*-
# creat_time: 2021/8/13 21:20

import numpy as np
from scipy.stats import norm


def CW_test(actual, forecast_1, forecast_2):
    """
    Performs the Clark and West (2007) test to compare forecasts from nested models.
    Reference:
    [1] T.E. Clark and K.D. West (2007). "Approximately Normal Tests
    for Equal Predictive Accuracy in Nested Models." Journal of
    Econometrics 138, 291-311
    [2] He M, Zhang Y, Wen D, et al. Forecasting crude oil prices:
    A scaled PCA approach[J]. Energy Economics, 2021, 97: 105189.

    :param actual:  a column vector of actual values
    :param forecast_1:  a column vector of forecasts for restricted model
    :param forecast_2:  a column vector of forecasts for unrestricted model
    :return: a tuple of two elements, the first element is the MSPE_adjusted
    statistic, while the second one is the corresponding p-value
    """
    e_1 = actual - forecast_1
    e_2 = actual - forecast_2
    f_hat = np.square(e_1) - (np.square(e_2) - np.square(forecast_1 - forecast_2))
    Y_f = f_hat
    X_f = np.ones(f_hat.shape[0]).reshape(-1, 1)
    beta_f = np.linalg.inv(X_f.transpose() @ X_f) * (X_f.transpose() @ Y_f)
    e_f = Y_f - X_f * beta_f
    sig2_e = (e_f.transpose() @ e_f) / (Y_f.shape[0] - 1)
    cov_beta_f = sig2_e * np.linalg.inv(X_f.transpose() @ X_f)
    MSPE_adjusted = beta_f / np.sqrt(cov_beta_f)
    p_value = 1 - norm.cdf(MSPE_adjusted)
    return MSPE_adjusted[0][0], p_value[0][0]

