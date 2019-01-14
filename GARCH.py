import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data, wb
import datetime
import time
import math
import fix_yahoo_finance as yf
yf.pdr_override()
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from contextlib import contextmanager
import sys, os
import warnings
warnings.simplefilter('ignore')
import holidays
import quandl
quandl.ApiConfig.api_key = "yLLXnsJQ6-1-eyzE6vUZ"
import matplotlib.pyplot as plt
%matplotlib inline

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def GARCH(timeseries):

    best_bic = np.inf
    best_order = None
    best_model = None
    best_param = None

    p_rng = range(4)
    q_rng = range(4)
    d_rng = range(2)

    # Try out the best model by comparing AIC scores
    print('Selecting best model...')
    with suppress_stdout():
        for i in p_rng:
            for j in q_rng:
                for d in d_rng:
                    for dist in ['Normal','Student t']:
                        try:
                            # GARCH model
                            tmp_mdl = arch_model(timeseries, vol = 'GARCH', p=i, o=d, q=j,dist=dist).fit()
                            tmp_bic = tmp_mdl.bic
                            tmp_param = tmp_mdl.params

                            # Update best GARCH model for stock volatility
                            if tmp_bic < best_bic:
                                best_bic = tmp_bic
                                best_order = (i, d, j)
                                best_model = tmp_mdl
                                best_param = tmp_param

                        except: continue
    #print('ARIMA bic: {:6.2f} | order: {}'.format(best_bic_mu, best_order_mu))
    print('GARCH : {:6.2f} | order: {}'.format(best_bic, best_order))

    return best_model, best_param
