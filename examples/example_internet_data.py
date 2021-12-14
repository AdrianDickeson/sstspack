import re
import time
import concurrent.futures
import os
import logging

import pandas as pd
from numpy import array, full, nan, inf, zeros, identity, where, amin, isnan
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

from sstspack import (
    DynamicLinearGaussianModel as DLGM,
    Utilities as utl,
    GaussianModelDesign as md,
    Fitting as fit,
)

import plot_figs
import latex_tables

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def min_element(AIC_scores):
    """"""
    AIC_scores = AIC_scores.copy()
    AIC_scores[isnan(AIC_scores)] = inf
    return where(AIC_scores == amin(AIC_scores))


def read_internet_data():
    """"""
    data = []
    fn = "data/internet.dat"
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                elements = elements[1:3]
                elements = [float(val) for val in elements]
                data.append(elements)
    data = array(data)
    data_df = pd.DataFrame(data)
    data_df.columns = ["Users", "Change"]
    data_df.index = range(1, len(data_df) + 1)
    return data_df


def get_ARMA_model_function(p, q):
    """"""

    def internet_ARMA_model_function(parameters, **kwargs):
        y_timeseries = kwargs["y_series"]
        diffuse_states = kwargs["diffuse_states"]

        model_design = md.get_ARMA_model_design(
            y_timeseries.index,
            parameters[1 : (p + 1)],
            parameters[(p + 1) :],
            full((1, 1), parameters[0]),
        )

        return DLGM(
            y_timeseries,
            model_design,
            diffuse_states=diffuse_states,
            validate_input=False,
        )

    return internet_ARMA_model_function


def get_ARMA_model_AIC(args):
    """"""
    p = args[0]
    q = args[1]
    y_timeseries = args[2]

    if p == 0 and q == 0:
        return nan

    parameter_count = p + q
    dimension = 1 + parameter_count
    state_count = max(p, 1 + q)
    diffuse_states = [True] * state_count

    initial_parameter_values = array([0.1] * dimension)
    parameter_bounds = [(0, inf)] + [(-1, 1) for _ in range(parameter_count)]
    parameter_names = (
        ["H"]
        + [f"phi_{idx+1}" for idx in range(p)]
        + [f"theta_{idx+1}" for idx in range(q)]
    )

    model_function = get_ARMA_model_function(p, q)

    start_time = time.time()
    try:
        res = fit.fit_model_max_likelihood(
            initial_parameter_values,
            parameter_bounds,
            model_function,
            y_timeseries,
            diffuse_states=diffuse_states,
            parameter_names=parameter_names,
        )
    except LinAlgError:
        return nan
    end_time = time.time()
    logger.debug(f"p:{p} q:{q} took {end_time-start_time:.2f} seconds")
    return res.akaike_information_criterion


def get_ARMA_AIC_model_values(y_timeseries):
    """"""
    result = full((6, 6), nan)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        parameter_list = []
        for p in range(6):
            for q in range(6):
                parameter_list.append(array([p, q, y_timeseries], dtype=object))

        results = executor.map(get_ARMA_model_AIC, parameter_list)

        for idx, model_AIC in enumerate(results):
            result[parameter_list[idx][0], parameter_list[idx][1]] = model_AIC

    return result


def main():
    logger.debug("Reading Internet time series data")
    y_timeseries = read_internet_data()["Change"]
    logger.debug("Creating missing values")
    missing_data = y_timeseries.copy()
    missing_idx = [5, 15, 25, 35, 45, 55, 65, 71, 72, 73, 74, 75, 85, 95]
    missing_data[missing_idx] = pd.NA

    logger.debug("Calculating AIC values")
    start_time = time.time()
    AIC_values = get_ARMA_AIC_model_values(y_timeseries)
    end_time = time.time()
    logger.debug(
        "Calculation complete, " + f"time taken {end_time-start_time:.2f} seconds"
    )
    best_model = min_element(AIC_values)
    logger.debug(f"Highest quality model - p: {best_model[0][0]} q: {best_model[1][0]}")

    logger.debug("Calculating AIC values with missing data")
    start_time = time.time()
    missing_AIC_values = get_ARMA_AIC_model_values(missing_data)
    end_time = time.time()
    logger.debug(
        "Calculation complete, " + f"time taken {end_time-start_time:.2f} seconds"
    )
    best_model = min_element(missing_AIC_values)
    logger.debug(f"Highest quality model - p: {best_model[0][0]} q: {best_model[1][0]}")

    logger.debug("Appending missing data to time series")
    extended_y_timeseries = y_timeseries.append(
        pd.Series(array([pd.NA] * 20), index=range(100, 120))
    )

    logger.debug("Fitting ARMA(1,1) state space model by maximum likelihood")
    initial_parameter_values = array([10, 0.7, 0.2])
    parameter_bounds = array([(0, inf), (0, 1), (-1, 1)])
    parameter_names = array(["Q", "phi", "theta"])

    internet_model_function = get_ARMA_model_function(1, 1)
    diffuse_states = [True, True]

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        internet_model_function,
        extended_y_timeseries,
        diffuse_states=diffuse_states,
        parameter_names=parameter_names,
    )
    end_time = time.time()
    logger.debug(
        "Maximum likelihood search complete. Time taken:- "
        + f"{end_time-start_time:.2f} seconds"
    )

    for idx, name in enumerate(res.parameter_names):
        logger.debug(
            f"Maximum likelihood {name}: {res.parameters[idx]:.4} "
            # + f"From textbook: {textbook_parameters[name]:.4}"
        )

    model = res.model.copy()
    model.filter()

    logger.info("Producing tables")
    latex_tables.latex_table81(AIC_values)
    latex_tables.latex_table82(missing_AIC_values)

    logger.info("Producing figures")
    plot_figs.plot_fig88(y_timeseries, missing_data)
    plot_figs.plot_fig89(model)


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.DEBUG)

    logger.addHandler(stream_handler)
    # fit.logger.addHandler(stream_handler)
    main()
    plt.show()
