'''
Created on 4 Sep 2020

@author: adriandickeson
'''

import re
import datetime as dt

import pandas as pd 
from numpy import array, zeros, ones, full, identity, hstack, dot, ravel, inf
import matplotlib.pyplot as plt

from sstspack import StateSpaceModel as SSM
import sstspack.modeldata as md
import sstspack.fitting as fit

import plot_figs 

def eom(date):
    result = date
    while result.month == date.month:
        result += dt.timedelta(1)
    return result - dt.timedelta(1)


def read_seatbelt_data():
    with open('data/Seatbelt.dat', 'r') as f:
        data = []
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace('\t', ' ')
                line = line.replace('\n', ' ')
                elements = re.split(' +', line)
                elements = elements[1:6]
                elements = [float(val) for val in elements]
                data.append(elements)
        data = array(data)
        data_df = pd.DataFrame(data)
        data_df.columns = ['Drivers', 'Front', 'Rear', 'Kilometres', 'Petrol Price']
        index = [eom(dt.date(year, month, 1))
                 for year in range(1969, 1985) for month in range(1, 13)]
        data_df.index = index
        return data_df

def get_bivariate_series(data_df):
    '''
    '''
    data = [array([[data_df.loc[idx, 'Front']], [data_df.loc[idx, 'Rear']]])
            for idx in data_df.index]
    result = pd.Series(data, index=data_df.index)
    return result


if __name__ == '__main__':
    data = read_seatbelt_data()
    y_series = data.Drivers
    plot_figs.plot_fig81(data)

    H = 3.41598e-3
    Q = 9.35852e-4
    sigma2_omega = full(6, 5.01096e-7)

    model_data1 = md.get_local_level_model_data(y_series.index, Q, H)
    model_data2 = md.get_frequency_domain_seasonal_model_data(y_series.index, 12,
                                                              sigma2_omega, H)

    model_data = md.combine_model_data([model_data1, model_data2])
    a0 = zeros((12, 1))
    P0 = identity(12)
    diffuse_states = [True] * 12

    model = SSM(y_series, model_data, a0, P0, diffuse_states)
    model.disturbance_smoother()

    plot_figs.plot_fig82(model)
    plot_figs.plot_fig83(model)
    plot_figs.plot_fig84(model)

    model_data3 = md.get_intervention_model_data(data.index, dt.date(1983,2,28))
    model_data4 = md.get_time_varying_regression_model_data(data.index,
                                                            data['Petrol Price'].to_frame(),
                                                            Q=zeros((1,1)), H=zeros((1,1)))

    model_data = md.combine_model_data([model_data1, model_data2,
                                        model_data3, model_data4])
    a0 = zeros((14,1))
    a0[12,0] = -0.23773
    a0[13,0] = -0.29140
    P0 = identity(14)
    P0[12,12] = 0
    P0[13,13] = 0
    diffuse_states = [True] * 12 + [False] * 2

    model = SSM(y_series, model_data, a0, P0, diffuse_states)
    model.disturbance_smoother()

    plot_figs.plot_fig85(model)

    plot_figs.plot_fig86(data)

    H = 1.e-4 * array([[5.206, 4.789], [4.789, 10.24]])
    Q_level = 1.e-5 * array([[4.970, 2.860], [2.860, 1.646]])
    Q_seasonal_list = [zeros((2,2))] * 6
    model_data1 = md.get_local_level_model_data(y_series.index, Q_level, H)
    model_data2 = md.get_frequency_domain_seasonal_model_data(y_series.index, 12,
                                                              Q_seasonal_list, zeros((2,2)))
    model_data3 = md.get_time_varying_regression_model_data(y_series.index,
                                                            data[['Kilometres', 'Petrol Price']],
                                                            Q=zeros((4,4)), H=zeros((2,2)))
    model_data4 = md.get_intervention_model_data(data.index, dt.date(1983,2,28),
                                                 H=zeros((2,2)), Q=zeros((2,2)))

    model_data = md.combine_model_data([model_data1, model_data2,
                                        model_data4, model_data3])

    y_series = get_bivariate_series(data[['Front', 'Rear']])

    state_len = model_data.Z[model_data.index[0]].shape[1]
    a0 = zeros((state_len,1))
    a0[-6] = -0.41557
    a0[-3] = -0.29140
    P0 = zeros((state_len, state_len))
    diffuse_states = [True] * (state_len - 6) + [False] * 6
    model = SSM(y_series, model_data, a0, P0, diffuse_states)
    model.disturbance_smoother()

    plot_figs.plot_fig87(model)

    plt.show()
