# -*- coding: utf-8 -*-
# creat_time: 2023/3/9 10:34

# for robust check with newly identified variables
import pandas as pd
import numpy as np
import statsmodels.api as sm
#

def ogap_detrend(y):
    t = len(y)
    x = pd.DataFrame(dict(t1=np.arange(1, t + 1), t2=np.arange(1, t + 1) ** 2))
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.resid


