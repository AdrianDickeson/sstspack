import logging
import time
from copy import copy

import pandas as pd
from numpy import full, array, inf, zeros
import matplotlib.pyplot as plt

from sstspack import (
    DynamicLinearGaussianModel as DLGM,
    GaussianModelDesign as md,
    Fitting as fit,
    Utilities as utl,
)
import plot_figs as pf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_y_timeseries():
    """"""
    y_timeseries = pd.read_csv("data/Nile.dat").iloc[:, 0]
    y_timeseries.index = range(1871, 1971)
    for idx in y_timeseries.index:
        y_timeseries[idx] = full((1, 1), y_timeseries[idx])
    return y_timeseries


def nile_local_level_model(parameters, *args, **kwargs):
    model_design = kwargs["model_design"]
    for idx in model_design.index:
        model_design.H[idx] = full((1, 1), parameters[0])
        model_design.Q[idx] = full((1, 1), parameters[1])
    model_design.filter_run = False

    y_timeseries = kwargs["y_series"]
    a_initial = kwargs["a_initial"]
    P_initial = kwargs["P_initial"]

    return DLGM(y_timeseries, model_design, a_initial, P_initial)


def main():
    logger.debug("Reading Nile time series data")
    y_timeseries = read_y_timeseries()

    logger.debug("Fitting local level state space model by maximum likelihood")
    initial_parameter_values = array([10000, 1000])
    parameter_bounds = array([(0, inf), (0, inf)])
    parameter_names = array(["H", "Q"])

    nile_model_function = nile_local_level_model
    nile_model_design = md.get_local_level_model_design(
        y_timeseries, initial_parameter_values[0], initial_parameter_values[1]
    )

    a_initial = zeros((1, 1))
    P_initial = full((1, 1), 1e7)

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        nile_model_function,
        y_timeseries,
        a_initial=a_initial,
        P_initial=P_initial,
        model_design=nile_model_design,
        parameter_names=parameter_names,
    )
    end_time = time.time()
    logger.debug(
        "Maximum likelihood search complete. Time taken:- "
        + f"{end_time-start_time:.2f} seconds"
    )

    H = 15099
    sigma2_eta = 1469.1
    logger.debug(f"Maximum likelihood H: {res.parameters[0]:.1f},\tExpected: {H:.0f}")
    logger.debug(
        f"Maximum likelihood sigma2_eta: {res.parameters[1]:.2f},\t"
        + f"Expected: {sigma2_eta:.1f}"
    )

    logger.debug("Analysing time series data")
    ssm = res.model.copy()
    ssm.disturbance_smoother()

    logger.debug("Analysing timeseries with missing data")
    missing_y_timeseries = y_timeseries.copy()
    missing_idx = list(range(20, 40)) + list(range(60, 80))
    missing_y_timeseries.iloc[missing_idx] = pd.NA

    missing_ssm = res.model.copy()
    missing_ssm.y = missing_y_timeseries
    missing_ssm.smoother()

    forecast_data = pd.Series([pd.NA] * 30, index=range(1971, 2001))
    forecast_y_timeseries = y_timeseries.append(forecast_data)
    model_df = md.get_local_level_model_design(
        forecast_y_timeseries.index, res.parameters[1], res.parameters[0]
    )
    forecast_ssm = DLGM(forecast_y_timeseries, model_df, a_initial, P_initial)
    forecast_ssm.filter()

    logger.debug("Running diagnostics\n" + pf.run_diagnostics(ssm))

    logger.info("Producing figures")
    pf.plot_fig21(ssm)
    pf.plot_fig22(ssm)
    pf.plot_fig23(ssm)
    pf.plot_fig24(ssm, ssm.simulate_smoother())
    pf.plot_fig25(missing_ssm)
    pf.plot_fig26(forecast_ssm)
    pf.plot_fig27(ssm)
    pf.plot_fig28(ssm)


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.DEBUG)

    logger.addHandler(stream_handler)
    fit.logger.addHandler(stream_handler)
    main()
    plt.show()
