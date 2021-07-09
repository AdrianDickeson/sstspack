import time

import pandas as pd
import matplotlib.pyplot as plt
from numpy import (
    log,
    exp,
    ones,
    identity,
    zeros,
    full,
    diag,
    array,
    inf,
    cos,
    sin,
    dot,
    set_printoptions,
)

import sstspack.GaussianModelDesign as md
from sstspack.ExtendedModelDesign import get_nonlinear_model_design
from sstspack.ExtendedDynamicModelClass import ExtendedDynamicModel as EKF
from sstspack.Utilities import identity_fn
import sstspack.fitting as fit

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
            + exp(parameters[0] + parameters[1] * state[0, 0]) * state[2, 0],
        )

    return Z_fn


def get_T_fn(T):
    """"""

    def T_fn(state):
        return dot(T, state)

    return T_fn


def get_R_fn(R):
    """"""

    def R_fn(state):
        return R

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


def log_pandas_series(series):
    """"""
    result = series.copy()
    for idx in result.index:
        result[idx] = log(result[idx])
    return result


def get_extended_model_design(parameters, model_template=None, y_series=None, dt=None):
    """"""
    T_trend = md.get_local_linear_trend_model_design(1, identity(2), ones((1, 1)))
    T_seasonal = md.get_frequency_domain_seasonal_model_design(
        1, 12, [identity(2)] * 6, 1
    )
    model_design = md.combine_model_design([T_trend, T_seasonal])

    m_trend = T_seasonal.Z[0].shape[1]
    m_seasonal = T_seasonal.Z[0].shape[0]

    T = model_design.loc[0, "T"]
    R = model_design.loc[0, "R"]
    Q = model_design.loc[0, "Q"]

    Q_full = identity(Q.shape[0])
    Q_full[0, 0] = 0
    Q_full[1, 1] = parameters[1]
    for idx in range(m_trend, m_seasonal + m_trend):
        Q_full[idx, idx] = parameters[2] / m_seasonal
    Q_fn = get_Q_fn(Q_full)

    Z_fn = get_Z_fn(parameters[-2:])
    T_fn = get_T_fn(T)
    R_fn = get_R_fn(R)

    H_fn = get_H_fn(full((1, 1), parameters[0]))

    return get_nonlinear_model_design(y_series, Z_fn, T_fn, R_fn, Q_fn, H_fn)


if __name__ == "__main__":
    set_printoptions(precision=2)
    y_series = read_uk_visits_abroad_data()
    ylog_series = log_pandas_series(y_series)

    extended_model_function = get_extended_model_design
    extended_model_design_template = get_extended_model_design(
        array([0, 0, 1]), y_series=y_series
    )

    initial_parameter_values = array([0.25] * 5)
    parameter_bounds = array([(0, inf), (0, inf), (0, inf), (-inf, inf), (-inf, inf)])
    parameter_names = array(["H", "Q_trend", "Q_seasonal" "c_0", "c_mu"])

    # Diffuse initialisation used - a0, P0 are ignored
    a0 = zeros((24, 1))
    P0 = identity(24)
    diffuse_states = [False] * 24

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        extended_model_function,
        y_series,
        a0,
        P0,
        diffuse_states,
        extended_model_design_template,
        parameter_names,
        model_class=EKF,
    )
    end_time = time.time()

    model = res.model
    model.smoother()

    print("Extended River Data")
    print("-------------------")
    print(res)
    print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    # plot_figs.plot_fig141(y_series, ylog_series)

    # plt.show()
    print("Finished")
