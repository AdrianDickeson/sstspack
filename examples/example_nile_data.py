'''
Created on 19 Mar 2020

@author: adriandickeson
'''

import pandas as pd
from numpy import full

from sstspack import StateSpaceModel as SSM, modeldata as md
import plot_figs as pf

if __name__ == '__main__':
    nile_data = pd.read_csv('data/Nile.dat').iloc[:,0]
    nile_data.index = range(1871, 1971)

    H = 15099
    sigma2_eta = 1469.1

    model_df = md.get_local_level_model_data(nile_data.index, sigma2_eta, H)
    a0 = full((1,1), 0)
    P0 = full((1,1), 10 ** 7)

    ssm = SSM(nile_data, model_df, a0, P0)
    ssm.disturbance_smoother()

    missing_nile_data = nile_data.copy()
    missing_idx = list(range(20, 40)) + list(range(60, 80))
    missing_nile_data.iloc[missing_idx] = pd.NA

    missing_ssm = SSM(missing_nile_data, model_df, a0, P0)
    missing_ssm.smoother()

    forecast_data = pd.Series([pd.NA]*30, index=range(1971,2001))
    forecast_nile_data = nile_data.append(forecast_data)
    model_df = md.get_local_level_model_data(forecast_nile_data.index, sigma2_eta, H)
    forecast_ssm = SSM(forecast_nile_data, model_df, a0, P0)
    forecast_ssm.filter()

    pf.run_diagnostics(ssm)

    pf.plot_fig21(ssm)
    pf.plot_fig22(ssm)
    pf.plot_fig23(ssm)
    pf.plot_fig24(ssm, ssm.simulate_smoother())
    pf.plot_fig25(missing_ssm)
    pf.plot_fig26(forecast_ssm)
    pf.plot_fig27(ssm)
    pf.plot_fig28(ssm)
