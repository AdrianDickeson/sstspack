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
    pi as PI,
    sin,
    dot,
    sqrt,
    hstack,
    set_printoptions,
)
import numpy as np
import scipy.stats as stats

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
        data_series[idx] = full((1, 1), float(data_series[idx]))

    return data_series


def get_Z_fn(parameters):
    """"""
    c_0 = parameters[0]
    c_mu = parameters[1]

    def Z_fn(state):
        return full(
            (1, 1),
            state[0, 0] + exp(c_0 + c_mu * state[0, 0]) * state[2, 0],
        )

    return Z_fn


def get_Z_prime_fn(parameters):
    """"""
    c_0 = parameters[0]
    c_mu = parameters[1]

    def Z_prime_fn(states):
        result = zeros(states.shape).T
        exp_term = exp(c_0 + c_mu * states[0, 0])

        result[0, 0] = 1 + c_mu * exp_term * states[2, 0]
        result[0, 2] = exp_term

        return result

    return Z_prime_fn


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


def get_Q_fn(Q):
    """"""

    def Q_fn(state):
        return Q

    return Q_fn


def get_H_fn(H):
    """"""

    def H_fn(state):
        return H

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
    T_seasonal = md.get_time_domain_seasonal_model_design(1, 12, 1, 1)
    model_design = md.combine_model_design([T_trend, T_seasonal])

    T = model_design.loc[0, "T"]
    R = model_design.loc[0, "R"]
    Q = model_design.loc[0, "Q"]
    H = full((1, 1), parameters[0])

    Q_full = identity(Q.shape[0])
    Q_full[0, 0] = 0
    Q_full[1, 1] = parameters[1]
    Q_full[2, 2] = 1
    # Q_full[1, 2] = Q_full[2, 1] = parameters[3] * sqrt(parameters[1] * parameters[2])
    Q_fn = get_Q_fn(Q_full)

    Z_fn = get_Z_fn(parameters[2:])
    Z_prime_fn = get_Z_prime_fn(parameters[2:])
    T_fn = get_T_fn(T)
    R_fn = get_R_fn(R)

    H_fn = get_H_fn(H)

    return get_nonlinear_model_design(
        y_series, Z_fn, T_fn, R_fn, Q_fn, H_fn, Z_prime_fn
    )


def main():
    # sigma_epsilon = 0.116
    # sigma_zeta = 0.00090
    # c_0 = -5.098
    # c_mu = 2.5e-4  # 0.0984
    # sigma_kappa = 0.00088
    # rho = 0.921
    # lambda_c = 2 * PI / 589

    sigma2_epsilon = 0.116
    sigma2_trend = 0.00088
    c_0 = -5.098
    c_mu = 0.00025

    set_printoptions(precision=2)
    y_series = read_uk_visits_abroad_data()
    ylog_series = log_pandas_series(y_series)

    initial_parameter_values = array([sigma2_epsilon, sigma2_trend, c_0, c_mu])
    extended_model_function = get_extended_model_design
    extended_model_design_template = get_extended_model_design(
        initial_parameter_values, y_series=y_series
    )

    parameter_bounds = array([(0, inf), (0, inf), (-6, 6), (-0.05, 0.05)])
    parameter_names = array(["H", "Q_trend", "c_0", "c_mu"])

    # Diffuse initialisation used - a0, P0 are ignored
    a0 = zeros((14, 1))
    P0 = 1e6 * identity(14)
    # diffuse_states = [True] * 14
    diffuse_states = [False] * 14

    parameters = array([sigma2_epsilon, sigma2_trend, c_0, c_mu])
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

    extended_model = res.model
    extended_model.smoother()

    print("Extended KF: UK Visitors Abroad Data")
    print("------------------------------------")
    print(res)
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")

    # extended_model_design = get_extended_model_design(parameters, None, y_series, None)

    # extended_model = EKF(y_series, extended_model_design, a0, P0, diffuse_states)
    # extended_model.smoother()

    plot_figs.plot_fig141(y_series, ylog_series)
    # plot_figs.plot_fig142(extended_model, c_0, c_mu)
    plot_figs.plot_fig142(extended_model, res.parameters[2], res.parameters[3])

    plt.show()
    print("Finished")


if __name__ == "__main__":
    main()
