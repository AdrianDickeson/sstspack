import re
import time
import concurrent.futures
import os

import pandas as pd
from numpy import array, full, nan, inf, zeros, identity
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

import sstspack.modeldata as md
import sstspack.fitting as fit

import plot_figs
import latex_tables


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
    return data_df


def get_ARMA_model_function(p, q):
    """"""

    def internet_ARMA_model_function(parameters, model_template):
        result = md.get_ARMA_model_data(
            model_template.index,
            parameters[1 : (p + 1)],
            parameters[(p + 1) :],
            full((1, 1), parameters[0]),
        )
        return result

    return internet_ARMA_model_function


def get_ARMA_model_AIC(args):  # p, q, y_timeseries):
    """"""
    p = args[0]
    q = args[1]
    y_timeseries = args[2]

    if p == 0 and q == 0:
        return nan

    parameter_count = p + q
    dimension = 1 + parameter_count
    state_count = max(p, 1 + q)

    initial_parameter_values = array([0.1] * dimension)
    parameter_bounds = [(0, inf)] + [(-1, 1) for _ in range(parameter_count)]
    parameter_names = (
        ["H"]
        + ["phi_{}".format(idx) for idx in range(p)]
        + ["theta_{}".format(idx) for idx in range(q)]
    )

    model_function = get_ARMA_model_function(p, q)
    model_template = model_function(initial_parameter_values, y_timeseries)

    a0 = zeros((state_count, 1))
    P0 = identity(state_count)
    diffuse_states = [True] * state_count

    start_time = time.time()
    try:
        res = fit.fit_model_max_likelihood(
            initial_parameter_values,
            parameter_bounds,
            model_function,
            y_timeseries,
            a0,
            P0,
            diffuse_states,
            model_template,
            parameter_names,
        )
    except LinAlgError:
        return nan
    end_time = time.time()
    print("p:{} q:{} took {:.2f} seconds".format(p, q, end_time - start_time))
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


if __name__ == "__main__":
    data = read_internet_data()["Change"]
    data.index = range(1, len(data) + 1)
    missing_data = data.copy()
    missing_idx = [5, 15, 25, 35, 45, 55, 65, 71, 72, 73, 74, 75, 85, 95]
    missing_data[missing_idx] = pd.NA

    plot_figs.plot_fig88(data, missing_data)

    AIC_values = get_ARMA_AIC_model_values(data)
    latex_tables.latex_table81(AIC_values)

    missing_AIC_values = get_ARMA_AIC_model_values(missing_data)
    latex_tables.latex_table82(missing_AIC_values)

    data = data.append(pd.Series(array([pd.NA] * 20), index=range(100, 120)))

    initial_parameter_values = array([10, 0.7, 0.2])
    parameter_bounds = array([(0, inf), (0, 1), (-1, 1)])
    parameter_names = array(["Q", "phi", "theta"])

    internet_model_function = get_ARMA_model_function(1, 1)
    internet_model_template = md.get_ARMA_model_data(
        data.index, [0.8], [0.2], full((1, 1), 10)
    )

    # Diffuse initialisation used - a0, P0 are ignored
    a0 = zeros((2, 1))
    P0 = 1e6 * identity(2)
    diffuse_states = [True, True]

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        internet_model_function,
        data,
        a0,
        P0,
        diffuse_states,
        internet_model_template,
        parameter_names,
    )
    end_time = time.time()

    model = res.model
    model.filter()
    print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    plot_figs.plot_fig89(model)

    plt.show()
