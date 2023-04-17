# -*- coding: utf-8 -*-
# creat_time: 2021/8/14 2:14

import numpy as np
from scipy.stats import norm

def PT_test(actual, forecast):
    """
    Implements the Directional Accuracy Test of Pesaran and Timmerman, 1992.
    Reference:
    Pesaran, M.H. and Timmermann, A. 1992, A simple nonparametric test of predictive performance,
    Journal of Business and Economic Statistics, 10(4), 461–465.

    :param actual: a column vector of actual values
    :param forecast: a column vector of the forecasted values.
    :return: a tuple of three elements, the first element is the success ratio,
    the second element is the PT statistic and the third one is the corresponding p-value.
    """
    n = actual.shape[0]
    if n != forecast.shape[0]:
        raise ValueError('Length of forecast and actual must be the same')
    x_t = np.zeros(n).reshape((-1, 1))
    z_t = np.zeros(n).reshape((-1, 1))
    y_t = np.zeros(n).reshape((-1, 1))
    x_t[actual > 0] = 1.0
    y_t[forecast > 0] = 1.0
    p_y = np.mean(y_t)
    p_x = np.mean(x_t)
    z_t[forecast * actual > 0] = 1
    p_hat = np.mean(z_t)
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
    p_hat_var = (p_star * (1 - p_star)) / n
    p_star_var = ((2 * p_y - 1) ** 2 * (p_x * (1 - p_x))) / n + \
                 ((2 * p_x - 1) ** 2 * (p_y * (1 - p_y))) / n + \
                 (4 * p_x * p_y * (1 - p_x) * (1 - p_y)) / n ** 2
    stat = (p_hat - p_star) / np.sqrt(p_hat_var - p_star_var)
    p_value = 1 - norm.cdf(stat)
    return p_hat, stat, p_value

