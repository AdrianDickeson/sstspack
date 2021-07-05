import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, exp, ones, identity, zeros, full, diag

from sstspack.ExtendedModelDesign import get_nonlinear_model_design
from sstspack.ExtendedDynamicModelClass import ExtendedDynamicModel as EKF
from sstspack.Utilities import identity_fn

import plot_figs


def read_uk_visits_abroad_data():
    """"""
    data_df = pd.read_csv("data/ott.csv")
    index_col = data_df.columns[0]
    data_col = data_df.columns[-3]

    data_series = data_df[data_col]
    data_series.index = data_df[index_col]

    start_date = "1980 JAN"
    end_date = "2006 DEC"
    start_index = data_series.index.tolist().index(start_date)
    end_index = data_series.index.tolist().index(end_date) + 1
    data_series = data_series[data_series.index[start_index:end_index]]

    for idx in data_series.index:
        data_series[idx] = float(data_series[idx])

    return data_series


def get_Z_fn(parameters):
    """"""

    def Z_fn(state):
        return full(
            (1, 1),
            state[0, 0]
            + exp(parameters[0] + parameters[1] * state[1, 0]) * state[2, 0],
        )

    return Z_fn


def get_T_fn():
    """"""

    def T_fn(state):
        return identity_fn(state)

    return T_fn


def get_R_fn(m):
    """"""

    def R_fn(state):
        return identity(m)

    return R_fn


def get_Q_fn(parameters):
    """"""

    def Q_fn(state):
        return diag(parameters)

    return Q_fn


def get_H_fn(parameters):
    """"""

    def H_fn(state):
        return diag(parameters)

    return H_fn


if __name__ == "__main__":
    y_series = read_uk_visits_abroad_data()
    ylog_series = y_series.copy()
    for idx in ylog_series.index:
        ylog_series[idx] = log(ylog_series[idx])

    Z_fn = get_Z_fn(ones(2))
    T_fn = get_T_fn()
    R_fn = get_R_fn(3)
    Q_fn = get_Q_fn(ones(3))
    H_fn = get_H_fn(ones(1))

    a_prior_initial = zeros((3, 1))
    P_prior_initial = identity(3)

    extended_model_design = get_nonlinear_model_design(
        y_series, Z_fn, T_fn, R_fn, Q_fn, H_fn
    )
    extended_model = EKF(
        y_series, extended_model_design, a_prior_initial, P_prior_initial
    )
    extended_model.filter()

    # plot_figs.plot_fig141(y_series, ylog_series)

    fig, ax = plt.subplots(1)
    ax.plot(extended_model.index, extended_model.Z)

    plt.show()
    print("Finished")
