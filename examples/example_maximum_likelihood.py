from numpy import full, zeros, inf, array, identity, dot
import pandas as pd
import time
import matplotlib.pyplot as plt

import sstspack.GaussianModelDesign as md
import sstspack.fitting as fit

from example_nile_data import read_nile_data, nile_local_level_model
from example_seatbelt_data import (
    read_seatbelt_data,
    seatbelt_seasonal_model,
    get_seatbelt_model_template,
)
from example_internet_data import read_internet_data, get_ARMA_model_function


def plot_model_summary(model, title, labels, field="a_hat"):
    """"""
    state_series = model.aggregate_field(field)
    non_diffuse_index = model.non_diffuse_index

    _, ax = plt.subplots(1)
    ax.scatter(model.index, model.y, label=labels[0])
    ax.plot(non_diffuse_index, state_series[non_diffuse_index], "r-", label=labels[1])
    ax.set_title(title)
    ax.legend()


def main():
    print("Maximum Likelihood Examples")
    print("---------------------------\n")

    ##################################################
    # Example - Nile Data - Local Level Model

    y_timeseries = read_nile_data()

    # Parameters [ H, Q ]
    initial_parameter_values = array([10000, 1000])
    parameter_bounds = array([(0, inf), (0, inf)])
    parameter_names = array(["H", "Q"])

    nile_model_function = nile_local_level_model
    nile_model_template = md.get_local_level_model_design(y_timeseries.index, 1, 1)

    # Diffuse initialisation used - a0, P0 are ignored
    a0 = zeros((1, 1))
    P0 = full((1, 1), 1e6)
    diffuse_states = [True]

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        nile_model_function,
        y_timeseries,
        a0,
        P0,
        diffuse_states,
        nile_model_template,
        parameter_names,
    )
    end_time = time.time()

    model = res.model
    model.smoother()
    plot_model_summary(
        model,
        "Nile River Data - Local Level Model",
        ["Raw Data", "Model State"],
        "a_prior",
    )

    print("Nile River Data")
    print("---------------")
    print(res)
    print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    ##################################################
    # Example - Seatbelt Data - Local Level Model

    seatbelt_data = read_seatbelt_data()
    y_timeseries = seatbelt_data.Drivers

    # Parameters [ H, Q_local, Q_seasonal ]
    initial_parameter_values = array([0.01, 0.01, 0.001])
    parameter_bounds = array([(0, inf), (0, inf), (0, inf)])
    parameter_names = array(["H", "Q_local", "Q_seasonal"])

    seatbelt_model_function = seatbelt_seasonal_model
    seatbelt_model_template = get_seatbelt_model_template(y_timeseries)

    a0 = zeros((12, 1))
    P0 = identity(12)
    diffuse_states = [True] * 12

    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        seatbelt_model_function,
        y_timeseries,
        a0,
        P0,
        diffuse_states,
        seatbelt_model_template,
        parameter_names,
    )
    end_time = time.time()

    model = res.model
    model.smoother()
    plot_model_summary(
        model,
        "Road Traffic Accident Data - Seasonal Local Level Model",
        ["Raw Data", "Model State"],
        "a_prior",
    )

    print("Road Traffic Accident Data")
    print("--------------------------")
    print(res)
    print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    ##################################################
    # Example - Internet Data - Local Level Model

    y_timeseries = read_internet_data()["Change"]

    # Parameters [ Q, phi, theta ]
    initial_parameter_values = array([10, 0.7, 0.2])
    parameter_bounds = array([(0, inf), (0, 1), (-1, 1)])
    parameter_names = array(["Q", "phi", "theta"])

    internet_model_function = get_ARMA_model_function(1, 1)
    internet_model_template = md.get_ARMA_model_design(
        y_timeseries.index, [0.8], [0.2], full((1, 1), 10)
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
        y_timeseries,
        a0,
        P0,
        diffuse_states,
        internet_model_template,
        parameter_names,
    )
    end_time = time.time()

    model = res.model
    model.smoother()
    plot_model_summary(
        model,
        "Internet Data - Local Level Model",
        ["Raw Data", "Model State"],
        "a_prior",
    )

    print("Internet Data")
    print("-------------")
    print(res)
    print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    plt.show()


if __name__ == "__main__":
    main()
