'''
Created on 11 Oct 2020

@author: adriandickeson
'''

from numpy import full, zeros, inf, array
import pandas as pd

import sstspack.modeldata as md
import sstspack.fitting as fit 

def local_model(params):
    H = full((1,1), params[0])
    Q_level = full((1,1), params[1])
    return md.get_local_level_model_data(len(y_series), Q_level, H)

# def model_function(params, y_series):
#     H = params[0]
#     Q_level = params[1]
#     sigma2_omega = full(6, params[2])
# 
#     model_data1 = md.get_local_level_model_data(len(y_series), Q_level, H)
#     model_data1.index = y_series.index
#     model_data2 = md.get_frequency_domain_seasonal_model_data(len(y_series), 12,
#                                                               sigma2_omega, H)
#     model_data2.index = y_series.index
# 
#     model_data = md.combine_model_data([model_data1, model_data2])
#     return model_data


if __name__ == '__main__':
    all_data = pd.read_csv('data/Nile.dat')
    y_series = all_data.iloc[:,0]

    params0 = array([1000,1000])
    params_bounds = array([(0,inf),(0,inf)])
    model_func = local_model
    a0 = zeros((1,1))
    P0 = full((1,1),1e6)
    diffuse_states = [False]
    res = fit.fit_model_max_likelihood(params0, params_bounds, model_func, y_series,
                                       a0, P0, diffuse_states)
    print(res)

#     params0 = [H, Q_level, sigma2_omega]
#     model_data = model_function(params0, y_series)
#     params_bounds = [(0,inf), (0,inf), (0,inf)]
#     res = fit.fit_model_max_likelihood(params0, params_bounds, model_function,
#                                        y_series, a0, P0, diffuse_states)
