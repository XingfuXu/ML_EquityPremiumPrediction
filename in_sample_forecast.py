# -*- coding: utf-8 -*-
# creat_time: 2021/11/13 17:39
## In-sample Model Building
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.stats import zscore
from Perform_Selection_IC import select_IC
import torch
from NN_models import Net3
#
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
#
from sklearn.metrics import mean_squared_error
from Perform_CW_test import CW_test
import matplotlib.pyplot as plt

predictor_df = pd.read_csv('result_predictor.csv')
predictor_df.head()

predictor_df.plot('month', 'log_equity_premium')
plt.show()

predictor0 = predictor_df.drop(['month', 'equity_premium'], axis=1)

# set the log equity premium 1-month ahead
predictor = np.concatenate([predictor0['log_equity_premium'][1:].values.reshape(-1, 1),
                            predictor0.iloc[0:(predictor0.shape[0] - 1), 1:]], axis=1)
N = predictor.shape[0]
n_cols = predictor.shape[1]

# Actual one-month ahead log equity premium
actual = predictor[:, [0]]

# Historical average forecasting
y_pred_HA = predictor0['log_equity_premium'].values[0:(predictor0.shape[0] - 1), ].cumsum() / np.arange(1, N + 1)
y_pred_HA = y_pred_HA.reshape(-1, 1)


## Machine learning methods used in GKX (2020)
# OLS, or the Kitchen sink model
OLS = LinearRegression()
OLS.fit(predictor[:, 1:], predictor[:, [0]])
y_pred_OLS = OLS.predict(predictor[:, 1:]).reshape(-1, 1)

# PLS
PLS2 = PLSRegression()   # n_components=2 is the default setting
PLS2.fit(predictor[:, 1:], predictor[:, [0]])
y_pred_PLS = PLS2.predict(predictor[:, 1:]).reshape(-1, 1)

# PCR
k_max_variables = 3
X_train = predictor[:, 1:]
pca = PCA()
pca.fit(zscore(X_train, axis=0, ddof=1))
X_train_new = pca.transform(zscore(X_train, axis=0, ddof=1))
k_components = select_IC(predictor[:, [0]], X_train_new[:, :k_max_variables], IC=3)
F_hat_selected = X_train_new[:, :k_components]
PCR = LinearRegression()
PCR.fit(F_hat_selected, predictor[:, [0]])
y_pred_PCR = PCR.predict(F_hat_selected).reshape(-1, 1)

# LASSO
LASSO = LassoCV(cv=10)
LASSO.fit(predictor[:, 1:], predictor[:, 0])
y_pred_LASSO = LASSO.predict(predictor[:, 1:]).reshape(-1, 1)

# Elastic Net (ENet)
ENet = ElasticNetCV(cv=10)
ENet.fit(predictor[:, 1:], predictor[:, 0])
y_pred_ENet = ENet.predict(predictor[:, 1:]).reshape(-1, 1)


# Gradient boosted regression trees (GBRT)
GBRT = GradientBoostingRegressor()
GBRT.fit(predictor[:, 1:], predictor[:, 0])
y_pred_GBRT = GBRT.predict(predictor[:, 1:]).reshape(-1, 1)


# Random Forest (RF)
RF = RandomForestRegressor(random_state=0)
RF.fit(predictor[:, 1:], predictor[:, 0])
y_pred_RF = RF.predict(predictor[:, 1:]).reshape(-1, 1)


# Neural Networks with SGD
torch.manual_seed(1)
np.random.seed(1)
NN3 = Net3(n_cols - 1, 32, 16, 8, 1)
optimizer = torch.optim.SGD(NN3.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()
X_train_tensor = torch.tensor(predictor[:, 1:], dtype=torch.float)
y_train_tensor = torch.tensor(predictor[:, [0]], dtype=torch.float)

losses = []
for i in range(1000):
    out = NN3(X_train_tensor)
    loss = loss_func(out, y_train_tensor)
    optimizer.zero_grad()
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

losses = np.array(losses)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.show()  # converged
y_pred_NN3 = NN3(X_train_tensor).detach().numpy()


## Other commonly used machine learning method
# Support Vector Regression (SVR)
SVR_model = SVR()
SVR_model.fit(predictor[:, 1:], predictor[:, 0])
y_pred_SVR = SVR_model.predict(predictor[:, 1:]).reshape(-1, 1)
# K Neighbors Regressor (KNR)
KNR = KNeighborsRegressor()
KNR.fit(predictor[:, 1:], predictor[:, 0])
y_pred_KNR = KNR.predict(predictor[:, 1:]).reshape(-1, 1)
# AdaBoost
AdaBoost = AdaBoostRegressor()
AdaBoost.fit(predictor[:, 1:], predictor[:, 0])
y_pred_AdaBoost = AdaBoost.predict(predictor[:, 1:]).reshape(-1, 1)
# XGBoost
XGBoost = XGBRegressor()
XGBoost.fit(predictor[:, 1:], predictor[:, 0])
y_pred_XGBoost = XGBoost.predict(predictor[:, 1:]).reshape(-1, 1)
# Combination
y_pred_combination = np.concatenate([y_pred_OLS, y_pred_PLS, y_pred_PCR, y_pred_LASSO, y_pred_ENet, 
                                     y_pred_GBRT, y_pred_RF, y_pred_NN3, y_pred_SVR, y_pred_KNR, 
                                     y_pred_AdaBoost, y_pred_XGBoost], axis=1).mean(axis=1).reshape(-1, 1)


## In-sample prediction results for all years (ALL)
# OLS
MSFE_HA = mean_squared_error(y_pred_HA, actual)
MSFE_OLS_ALL = mean_squared_error(y_pred_OLS, actual)
IS_R_OLS_ALL = 1 - MSFE_OLS_ALL / MSFE_HA
MSFE_adjusted_OLS_ALL, p_OLS_ALL = CW_test(actual, y_pred_HA, y_pred_OLS)
# PLS
MSFE_PLS_ALL = mean_squared_error(y_pred_PLS, actual)
IS_R_PLS_ALL = 1 - MSFE_PLS_ALL / MSFE_HA
MSFE_adjusted_PLS_ALL, p_PLS_ALL = CW_test(actual, y_pred_HA, y_pred_PLS)
# PCR
MSFE_PCR_ALL = mean_squared_error(y_pred_PCR, actual)
IS_R_PCR_ALL = 1 - MSFE_PCR_ALL / MSFE_HA
MSFE_adjusted_PCR_ALL, p_PCR_ALL = CW_test(actual, y_pred_HA, y_pred_PCR)
# ENet
MSFE_ENet_ALL = mean_squared_error(y_pred_ENet, actual)
IS_R_ENet_ALL = 1 - MSFE_ENet_ALL / MSFE_HA
MSFE_adjusted_ENet_ALL, p_ENet_ALL = CW_test(actual, y_pred_HA, y_pred_ENet)
# LASSO
MSFE_LASSO_ALL = mean_squared_error(y_pred_LASSO, actual)
IS_R_LASSO_ALL = 1 - MSFE_LASSO_ALL / MSFE_HA
MSFE_adjusted_LASSO_ALL, p_LASSO_ALL = CW_test(actual, y_pred_HA, y_pred_LASSO)
# GBRT
MSFE_GBRT_ALL = mean_squared_error(y_pred_GBRT, actual)
IS_R_GBRT_ALL = 1 - MSFE_GBRT_ALL / MSFE_HA
MSFE_adjusted_GBRT_ALL, p_GBRT_ALL = CW_test(actual, y_pred_HA, y_pred_GBRT)
# RF
MSFE_RF_ALL = mean_squared_error(y_pred_RF, actual)
IS_R_RF_ALL = 1 - MSFE_RF_ALL / MSFE_HA
MSFE_adjusted_RF_ALL, p_RF_ALL = CW_test(actual, y_pred_HA, y_pred_RF)
# NN3
MSFE_NN3_ALL = mean_squared_error(y_pred_NN3, actual)
IS_R_NN3_ALL = 1 - MSFE_NN3_ALL / MSFE_HA
MSFE_adjusted_NN3_ALL, p_NN3_ALL = CW_test(actual, y_pred_HA, y_pred_NN3)
# Other commonly used ML methods
# SVR
MSFE_SVR_ALL = mean_squared_error(y_pred_SVR, actual)
IS_R_SVR_ALL = 1 - MSFE_SVR_ALL / MSFE_HA
MSFE_adjusted_SVR_ALL, p_SVR_ALL = CW_test(actual, y_pred_HA, y_pred_SVR)
# KNR
MSFE_KNR_ALL = mean_squared_error(y_pred_KNR, actual)
IS_R_KNR_ALL = 1 - MSFE_KNR_ALL / MSFE_HA
MSFE_adjusted_KNR_ALL, p_KNR_ALL = CW_test(actual, y_pred_HA, y_pred_KNR)
# AdaBoost
MSFE_AdaBoost_ALL = mean_squared_error(y_pred_AdaBoost, actual)
IS_R_AdaBoost_ALL = 1 - MSFE_AdaBoost_ALL / MSFE_HA
MSFE_adjusted_AdaBoost_ALL, p_AdaBoost_ALL = CW_test(actual, y_pred_HA, y_pred_AdaBoost)
# XGBoost
MSFE_XGBoost_ALL = mean_squared_error(y_pred_XGBoost, actual)
IS_R_XGBoost_ALL = 1 - MSFE_XGBoost_ALL / MSFE_HA
MSFE_adjusted_XGBoost_ALL, p_XGBoost_ALL = CW_test(actual, y_pred_HA, y_pred_XGBoost)
# Combination
MSFE_combination_ALL = mean_squared_error(y_pred_combination, actual)
IS_R_combination_ALL = 1 - MSFE_combination_ALL / MSFE_HA
MSFE_adjusted_combination_ALL, p_combination_ALL = CW_test(actual, y_pred_HA, y_pred_combination)


## Prediction results for forecasts begin at 1957:01
in_out_1957 = predictor_df.index[predictor_df['month'] == 195701][0]
# OLS
MSFE_HA = mean_squared_error(y_pred_HA[in_out_1957:, ], actual[in_out_1957:, ])
MSFE_OLS_1957 = mean_squared_error(y_pred_OLS[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_OLS_1957 = 1 - MSFE_OLS_1957 / MSFE_HA
MSFE_adjusted_OLS_1957, p_OLS_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_OLS[in_out_1957:, ])
# PLS
MSFE_PLS_1957 = mean_squared_error(y_pred_PLS[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_PLS_1957 = 1 - MSFE_PLS_1957 / MSFE_HA
MSFE_adjusted_PLS_1957, p_PLS_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_PLS[in_out_1957:, ])
# PCR
MSFE_PCR_1957 = mean_squared_error(y_pred_PCR[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_PCR_1957 = 1 - MSFE_PCR_1957 / MSFE_HA
MSFE_adjusted_PCR_1957, p_PCR_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_PCR[in_out_1957:, ])
# ENet
MSFE_ENet_1957 = mean_squared_error(y_pred_ENet[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_ENet_1957 = 1 - MSFE_ENet_1957 / MSFE_HA
MSFE_adjusted_ENet_1957, p_ENet_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_ENet[in_out_1957:, ])
# LASSO
MSFE_LASSO_1957 = mean_squared_error(y_pred_LASSO[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_LASSO_1957 = 1 - MSFE_LASSO_1957 / MSFE_HA
MSFE_adjusted_LASSO_1957, p_LASSO_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_LASSO[in_out_1957:, ])
# GBRT
MSFE_GBRT_1957 = mean_squared_error(y_pred_GBRT[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_GBRT_1957 = 1 - MSFE_GBRT_1957 / MSFE_HA
MSFE_adjusted_GBRT_1957, p_GBRT_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_GBRT[in_out_1957:, ])
# RF
MSFE_RF_1957 = mean_squared_error(y_pred_RF[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_RF_1957 = 1 - MSFE_RF_1957 / MSFE_HA
MSFE_adjusted_RF_1957, p_RF_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_RF[in_out_1957:, ])
# NN3
MSFE_NN3_1957 = mean_squared_error(y_pred_NN3[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_NN3_1957 = 1 - MSFE_NN3_1957 / MSFE_HA
MSFE_adjusted_NN3_1957, p_NN3_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_NN3[in_out_1957:, ])
## Other commonly used ML methods
# SVR
MSFE_SVR_1957 = mean_squared_error(y_pred_SVR[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_SVR_1957 = 1 - MSFE_SVR_1957 / MSFE_HA
MSFE_adjusted_SVR_1957, p_SVR_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_SVR[in_out_1957:, ])
# KNR
MSFE_KNR_1957 = mean_squared_error(y_pred_KNR[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_KNR_1957 = 1 - MSFE_KNR_1957 / MSFE_HA
MSFE_adjusted_KNR_1957, p_KNR_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_KNR[in_out_1957:, ])
# AdaBoost
MSFE_AdaBoost_1957 = mean_squared_error(y_pred_AdaBoost[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_AdaBoost_1957 = 1 - MSFE_AdaBoost_1957 / MSFE_HA
MSFE_adjusted_AdaBoost_1957, p_AdaBoost_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_AdaBoost[in_out_1957:, ])
# XGBoost
MSFE_XGBoost_1957 = mean_squared_error(y_pred_XGBoost[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_XGBoost_1957 = 1 - MSFE_XGBoost_1957 / MSFE_HA
MSFE_adjusted_XGBoost_1957, p_XGBoost_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_XGBoost[in_out_1957:, ])
# Combination
MSFE_combination_1957 = mean_squared_error(y_pred_combination[in_out_1957:, ], actual[in_out_1957:, ])
IS_R_combination_1957 = 1 - MSFE_combination_1957 / MSFE_HA
MSFE_adjusted_combination_1957, p_combination_1957 = CW_test(actual[in_out_1957:, ], y_pred_HA[in_out_1957:, ], y_pred_combination[in_out_1957:, ])

## Prediction results for forecasts begin at 1990:01
in_out_1990 = predictor_df.index[predictor_df['month'] == 199001][0]
# OLS
MSFE_HA = mean_squared_error(y_pred_HA[in_out_1990:, ], actual[in_out_1990:, ])
MSFE_OLS_1990 = mean_squared_error(y_pred_OLS[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_OLS_1990 = 1 - MSFE_OLS_1990 / MSFE_HA
MSFE_adjusted_OLS_1990, p_OLS_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_OLS[in_out_1990:, ])
# PLS
MSFE_PLS_1990 = mean_squared_error(y_pred_PLS[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_PLS_1990 = 1 - MSFE_PLS_1990 / MSFE_HA
MSFE_adjusted_PLS_1990, p_PLS_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_PLS[in_out_1990:, ])
# PCR
MSFE_PCR_1990 = mean_squared_error(y_pred_PCR[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_PCR_1990 = 1 - MSFE_PCR_1990 / MSFE_HA
MSFE_adjusted_PCR_1990, p_PCR_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_PCR[in_out_1990:, ])
# ENet
MSFE_ENet_1990 = mean_squared_error(y_pred_ENet[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_ENet_1990 = 1 - MSFE_ENet_1990 / MSFE_HA
MSFE_adjusted_ENet_1990, p_ENet_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_ENet[in_out_1990:, ])
# LASSO
MSFE_LASSO_1990 = mean_squared_error(y_pred_LASSO[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_LASSO_1990 = 1 - MSFE_LASSO_1990 / MSFE_HA
MSFE_adjusted_LASSO_1990, p_LASSO_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_LASSO[in_out_1990:, ])
# GBRT
MSFE_GBRT_1990 = mean_squared_error(y_pred_GBRT[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_GBRT_1990 = 1 - MSFE_GBRT_1990 / MSFE_HA
MSFE_adjusted_GBRT_1990, p_GBRT_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_GBRT[in_out_1990:, ])
# RF
MSFE_RF_1990 = mean_squared_error(y_pred_RF[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_RF_1990 = 1 - MSFE_RF_1990 / MSFE_HA
MSFE_adjusted_RF_1990, p_RF_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_RF[in_out_1990:, ])
# NN3
MSFE_NN3_1990 = mean_squared_error(y_pred_NN3[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_NN3_1990 = 1 - MSFE_NN3_1990 / MSFE_HA
MSFE_adjusted_NN3_1990, p_NN3_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_NN3[in_out_1990:, ])
## other commonly used ML methods
# SVR
MSFE_SVR_1990 = mean_squared_error(y_pred_SVR[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_SVR_1990 = 1 - MSFE_SVR_1990 / MSFE_HA
MSFE_adjusted_SVR_1990, p_SVR_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_SVR[in_out_1990:, ])
# KNR
MSFE_KNR_1990 = mean_squared_error(y_pred_KNR[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_KNR_1990 = 1 - MSFE_KNR_1990 / MSFE_HA
MSFE_adjusted_KNR_1990, p_KNR_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_KNR[in_out_1990:, ])
# AdaBoost
MSFE_AdaBoost_1990 = mean_squared_error(y_pred_AdaBoost[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_AdaBoost_1990 = 1 - MSFE_AdaBoost_1990 / MSFE_HA
MSFE_adjusted_AdaBoost_1990, p_AdaBoost_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_AdaBoost[in_out_1990:, ])
# XGBoost
MSFE_XGBoost_1990 = mean_squared_error(y_pred_XGBoost[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_XGBoost_1990 = 1 - MSFE_XGBoost_1990 / MSFE_HA
MSFE_adjusted_XGBoost_1990, p_XGBoost_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_XGBoost[in_out_1990:, ])
# Combination
MSFE_combination_1990 = mean_squared_error(y_pred_combination[in_out_1990:, ], actual[in_out_1990:, ])
IS_R_combination_1990 = 1 - MSFE_combination_1990 / MSFE_HA
MSFE_adjusted_combination_1990, p_combination_1990 = CW_test(actual[in_out_1990:, ], y_pred_HA[in_out_1990:, ], y_pred_combination[in_out_1990:, ])


#
results_in_sample_forecast1 = np.array([
    [IS_R_OLS_ALL, MSFE_adjusted_OLS_ALL, p_OLS_ALL, IS_R_OLS_1957, MSFE_adjusted_OLS_1957, p_OLS_1957, 
     IS_R_OLS_1990, MSFE_adjusted_OLS_1990, p_OLS_1990],
    [IS_R_PLS_ALL, MSFE_adjusted_PLS_ALL, p_PLS_ALL, IS_R_PLS_1957, MSFE_adjusted_PLS_1957, p_PLS_1957,
     IS_R_PLS_1990, MSFE_adjusted_PLS_1990, p_PLS_1990],
    [IS_R_PCR_ALL, MSFE_adjusted_PCR_ALL, p_PCR_ALL, IS_R_PCR_1957, MSFE_adjusted_PCR_1957, p_PCR_1957,
     IS_R_PCR_1990, MSFE_adjusted_PCR_1990, p_PCR_1990],
    [IS_R_LASSO_ALL, MSFE_adjusted_LASSO_ALL, p_LASSO_ALL, IS_R_LASSO_1957, MSFE_adjusted_LASSO_1957, p_LASSO_1957,
     IS_R_LASSO_1990, MSFE_adjusted_LASSO_1990, p_LASSO_1990],
    [IS_R_ENet_ALL, MSFE_adjusted_ENet_ALL, p_ENet_ALL, IS_R_ENet_1957, MSFE_adjusted_ENet_1957, p_ENet_1957,
     IS_R_ENet_1990, MSFE_adjusted_ENet_1990, p_ENet_1990],
    [IS_R_GBRT_ALL, MSFE_adjusted_GBRT_ALL, p_GBRT_ALL, IS_R_GBRT_1957, MSFE_adjusted_GBRT_1957, p_GBRT_1957,
     IS_R_GBRT_1990, MSFE_adjusted_GBRT_1990, p_GBRT_1990],
    [IS_R_RF_ALL, MSFE_adjusted_RF_ALL, p_RF_ALL, IS_R_RF_1957, MSFE_adjusted_RF_1957, p_RF_1957,
     IS_R_RF_1990, MSFE_adjusted_RF_1990, p_RF_1990],
    [IS_R_NN3_ALL, MSFE_adjusted_NN3_ALL, p_NN3_ALL, IS_R_NN3_1957, MSFE_adjusted_NN3_1957, p_NN3_1957,
     IS_R_NN3_1990, MSFE_adjusted_NN3_1990, p_NN3_1990]
])
results_in_sample_forecast1 = pd.DataFrame(results_in_sample_forecast1)
results_in_sample_forecast1.insert(0, "Forecasting models",  ["OLS", "PLS", "PCR", "LASSO", "ENet",
                                                              "GBRT", "RF", "NN3"])
results_in_sample_forecast1.to_csv("results_in_sample_forecast1.csv", index=False)
#
results_in_sample_forecast2 = np.array([
    [IS_R_SVR_ALL, MSFE_adjusted_SVR_ALL, p_SVR_ALL, IS_R_SVR_1957, MSFE_adjusted_SVR_1957, p_SVR_1957, 
     IS_R_SVR_1990, MSFE_adjusted_SVR_1990, p_SVR_1990],
    [IS_R_KNR_ALL, MSFE_adjusted_KNR_ALL, p_KNR_ALL, IS_R_KNR_1957, MSFE_adjusted_KNR_1957, p_KNR_1957,
     IS_R_KNR_1990, MSFE_adjusted_KNR_1990, p_KNR_1990],
    [IS_R_AdaBoost_ALL, MSFE_adjusted_AdaBoost_ALL, p_AdaBoost_ALL, IS_R_AdaBoost_1957, MSFE_adjusted_AdaBoost_1957, p_AdaBoost_1957,
     IS_R_AdaBoost_1990, MSFE_adjusted_AdaBoost_1990, p_AdaBoost_1990],
    [IS_R_XGBoost_ALL, MSFE_adjusted_XGBoost_ALL, p_XGBoost_ALL, IS_R_XGBoost_1957, MSFE_adjusted_XGBoost_1957, p_XGBoost_1957,
     IS_R_XGBoost_1990, MSFE_adjusted_XGBoost_1990, p_XGBoost_1990],
    [IS_R_combination_ALL, MSFE_adjusted_combination_ALL, p_combination_ALL, IS_R_combination_1957, MSFE_adjusted_combination_1957, p_combination_1957,
     IS_R_combination_1990, MSFE_adjusted_combination_1990, p_combination_1990]
])
results_in_sample_forecast2 = pd.DataFrame(results_in_sample_forecast2)
results_in_sample_forecast2.insert(0, "Forecasting models",
                                   ["SVR", "KNR", "AdaBoost", "XGBoost", "Combination"])
results_in_sample_forecast2.to_csv("results_in_sample_forecast2.csv", index=False)
#
