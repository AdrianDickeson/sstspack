import re
import datetime as dt
import logging
import time

import pandas as pd
from numpy import (
    exp,
    array,
    zeros,
    ones,
    full,
    identity,
    hstack,
    dot,
    ravel,
    inf,
    diag,
    sqrt,
)
import matplotlib.pyplot as plt

from sstspack import (
    DynamicLinearGaussianModelClass as DLGMC,
    DynamicLinearGaussianModel as DLGM,
    Fitting as fit,
    Utilities as utl,
    GaussianModelDesign as md,
)

import plot_figs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def eom(date):
    """"""
    result = date
    while result.month == date.month:
        result += dt.timedelta(1)
    return result - dt.timedelta(1)


def read_seatbelt_data():
    """"""
    with open("data/Seatbelt.dat", "r") as f:
        data = []
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                elements = elements[1:6]
                elements = [float(val) for val in elements]
                data.append(elements)
        data = array(data)
        data_df = pd.DataFrame(data)
        data_df.columns = ["Drivers", "Front", "Rear", "Kilometres", "Petrol Price"]
        index = [
            eom(dt.date(year, month, 1))
            for year in range(1969, 1985)
            for month in range(1, 13)
        ]
        data_df.index = index
        return data_df


def get_bivariate_series(data_df):
    """"""
    data = [
        array([[data_df.loc[idx, "Front"]], [data_df.loc[idx, "Rear"]]])
        for idx in data_df.index
    ]
    return pd.Series(data, index=data_df.index)


def get_seatbelt_model_template(y_timeseries):
    H = 1
    Q = 1
    sigma2_omega = full(6, 1)

    model_data1 = md.get_local_level_model_design(y_timeseries.index, Q, H)
    model_data2 = md.get_frequency_domain_seasonal_model_design(
        y_timeseries.index, 12, sigma2_omega, H
    )

    return md.combine_model_design([model_data1, model_data2])


def seatbelt_seasonal_model(parameters, *args, **kwargs):
    """"""
    H = full((1, 1), parameters[0])
    Q_local = parameters[1]
    Q_seasonal = parameters[2]
    Q_full = identity(12)
    Q_full[0, 0] = Q_local
    for idx in range(1, 12):
        Q_full[idx, idx] = Q_seasonal

    model_design = kwargs["model_design"]
    for idx in model_design.index:
        model_design.H[idx] = H
        model_design.Q[idx] = Q_full

    y_timeseries = kwargs["y_series"]
    diffuse_states = kwargs["diffuse_states"]

    return DLGM(
        y_timeseries, model_design, diffuse_states=diffuse_states, validate_input=False
    )


def bivariate_seatbelt_model(parameters, **kwargs):
    """"""
    data = kwargs["timeseries_data"]
    y_timeseries = kwargs["y_series"]
    H = diag(parameters[:2])
    H[1, 0] = H[0, 1] = parameters[2] * sqrt(parameters[0] * parameters[1])
    Q_level = diag(parameters[3:5])
    Q_level[1, 0] = Q_level[0, 1] = parameters[5] * sqrt(parameters[3] * parameters[4])
    Q_seasonal_list = [zeros((2, 2))] * 6

    model_design1 = md.get_local_level_model_design(y_timeseries.index, Q_level, H)
    model_design2 = md.get_frequency_domain_seasonal_model_design(
        y_timeseries.index, 12, Q_seasonal_list, zeros((2, 2))
    )
    model_design3 = md.get_time_varying_regression_model_design(
        y_timeseries.index,
        data[["Kilometres", "Petrol Price"]],
        Q=zeros((4, 4)),
        H=zeros((2, 2)),
    )
    model_design4 = md.get_intervention_model_design(
        y_timeseries.index, dt.date(1983, 2, 28), H=zeros((2, 2)), Q=zeros((2, 2))
    )

    combine_model_design = md.combine_model_design(
        [model_design1, model_design2, model_design4, model_design3]
    )

    a_initial = kwargs["a_initial"]
    P_initial = kwargs["P_initial"]
    diffuse_states = kwargs["diffuse_states"]

    return DLGM(
        y_timeseries,
        combine_model_design,
        a_initial,
        P_initial,
        diffuse_states,
        validate_input=False,
    )


def main():
    logger.debug("Reading Seatbelt timeseries data")
    data = read_seatbelt_data()
    y_timeseries = data.Drivers

    logger.debug("Fitting seasonal state space model using maximum likelihood")
    initial_parameter_values = array([0.01, 0.01, 0.001])
    parameter_bounds = array([(0, inf), (0, inf), (0, inf)])
    parameter_names = array(["H", "Q_local", "Q_seasonal"])

    seatbelt_model_function = seatbelt_seasonal_model
    seasonal_model_design = get_seatbelt_model_template(y_timeseries)

    diffuse_states = [True] * 12

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        seatbelt_model_function,
        y_timeseries,
        model_design=seasonal_model_design,
        diffuse_states=diffuse_states,
        parameter_names=parameter_names,
    )
    end_time = time.time()
    logger.debug(
        "Maximum likelihood search complete. Time taken:- "
        + f"{end_time-start_time:.2f} seconds"
    )

    textbook_parameters = {
        "H": 3.41598e-3,
        "Q_local": 9.35852e-4,
        "Q_seasonal": 5.01096e-7,
    }
    for idx, name in enumerate(res.parameter_names):
        logger.debug(
            f"Maximum likelihood {name}: {res.parameters[idx]:.4} "
            + f"From textbook: {textbook_parameters[name]:.4}"
        )

    model_data1 = md.get_local_level_model_design(
        y_timeseries.index, res.parameters[1], res.parameters[0]
    )
    model_data2 = md.get_frequency_domain_seasonal_model_design(
        y_timeseries.index, 12, full(6, res.parameters[2]), res.parameters[0]
    )

    seasonal_model = res.model.copy()
    logger.debug("Analysing time series data using seasonal model")
    seasonal_model.disturbance_smoother()

    model_data3 = md.get_time_varying_regression_model_design(
        data.index, data["Petrol Price"].to_frame(), Q=zeros((1, 1)), H=zeros((1, 1))
    )
    model_data4 = md.get_intervention_model_design(data.index, dt.date(1983, 2, 28))

    model_data = md.combine_model_design(
        [model_data1, model_data2, model_data3, model_data4]
    )

    a_initial = zeros((14, 1))
    P_initial = 1e7 * identity(14)
    regression_coefficient = -0.29140
    seatbelt_law_effect = -0.23773
    diffuse_states = [True] * 13 + [False]

    combined_model = DLGM(
        y_timeseries, model_data, a_initial, P_initial, diffuse_states=diffuse_states
    )
    logger.debug(
        "Analysing time series data with seasonal model with intervention "
        + "and regression effects"
    )
    combined_model.disturbance_smoother()
    logger.debug(
        "Estimated regression coefficient: "
        + f"{combined_model.a_hat[combined_model.index[-1]][12,0]:.5}\tExpected: "
        + f"{regression_coefficient:.5}"
    )
    logger.debug(
        "Estimated effect of Seatbelt law: "
        + f"{combined_model.a_hat[combined_model.index[-1]][13,0]:.2f} "
        + f"({100*(exp(combined_model.a_hat[combined_model.index[-1]][13,0])-1):.2f}%)"
        + f"\tExpected: {seatbelt_law_effect:.2f}"
    )

    y_timeseries = get_bivariate_series(data[["Front", "Rear"]])

    logger.debug("Fitting bivariate state space model using maximum likelihood")
    state_len = 30  # bivariate_model_design.Z[bivariate_model_design.index[0]].shape[1]

    a_initial = zeros((state_len, 1))
    a_initial[-6] = -0.41557
    a_initial[-3] = -0.29140
    P_initial = zeros((state_len, state_len))

    diffuse_states = [True] * (state_len - 6) + [False] * 6

    initial_parameter_values = [1e-3, 1e-3, 0, 1e-4, 1e-4, 0]
    bivariate_seatbelt_model_function = bivariate_seatbelt_model
    parameter_bounds = [(0, inf), (0, inf), (-1, 1), (0, inf), (0, inf), (-1, 1)]
    parameter_names = ["H_front", "H_rear", "rho_H", "Q_front", "Q_rear", "rho_Q"]

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        bivariate_seatbelt_model_function,
        y_timeseries,
        a_initial=a_initial,
        P_initial=P_initial,
        diffuse_states=diffuse_states,
        timeseries_data=data,
        parameter_names=parameter_names,
    )
    end_time = time.time()
    bivariate_model = res.model
    logger.debug(
        "Maximum likelihood search complete. Time taken:- "
        + f"{end_time-start_time:.2f} seconds"
    )

    for idx, name in enumerate(res.parameter_names):
        logger.debug(
            f"Maximum likelihood {name}: {res.parameters[idx]:.4} "
            # + f"From textbook: {textbook_parameters[name]:.4}"
        )

    logger.debug("Analysing bivariate time series data")
    bivariate_model.disturbance_smoother()

    logger.info("Producing figures")
    plot_figs.plot_fig81(data)
    plot_figs.plot_fig82(seasonal_model)
    plot_figs.plot_fig83(seasonal_model)
    plot_figs.plot_fig84(seasonal_model)
    plot_figs.plot_fig85(combined_model)
    plot_figs.plot_fig86(data)
    plot_figs.plot_fig87(bivariate_model)


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.DEBUG)

    logger.addHandler(stream_handler)
    # DLGMC.logger.addHandler(stream_handler)
    main()
    plt.show()
