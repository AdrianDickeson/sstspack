from numpy import full, zeros, inf, array, identity
import pandas as pd
import time

import sstspack.modeldata as md
import sstspack.fitting as fit

from example_seatbelt_data import read_seatbelt_data


# def nile_local_local_model2(parameters, model_template):
#     H = full((1, 1), parameters[0])
#     Q = full((1, 1), parameters[1])
#     return md.get_local_level_model_data(len(y_timeseries), Q, H)


def nile_local_level_model(parameters, model_template):
    model_template.H = parameters[0]
    model_template.Q = parameters[1]
    return model_template


def get_seatbelt_model_template(y_timeseries):
    H = 1
    Q = 1
    sigma2_omega = full(6, 1)

    model_data1 = md.get_local_level_model_data(y_timeseries.index, Q, H)
    model_data2 = md.get_frequency_domain_seasonal_model_data(
        y_timeseries.index, 12, sigma2_omega, H
    )

    model_template = md.combine_model_data([model_data1, model_data2])
    return model_template


def seatbelt_seasonal_model(parameters, model_template):
    H = parameters[0]
    Q_local = parameters[1]
    Q_seasonal = parameters[2]
    Q_full = identity(12)
    Q_full[0, 0] = Q_local
    for idx in range(1, 12):
        Q_full[idx, idx] = Q_seasonal

    for idx in model_template.index:
        model_template.H[idx] = H
        model_template.Q[idx] = Q_full

    return model_template


if __name__ == "__main__":
    print("Maximum Likelihood Examples")
    print("---------------------------\n")

    ##################################################
    # Example - Nile Data - Local Level Model

    nile_data = pd.read_csv("data/Nile.dat")
    y_timeseries = nile_data.iloc[:, 0]

    # Parameters [ H, Q ]
    initial_parameter_values = array([10000, 1000])
    parameter_bounds = array([(0, inf), (0, inf)])

    nile_model_function = nile_local_level_model
    nile_model_template = md.get_local_level_model_data(y_timeseries.index, 1, 1)

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
    )
    end_time = time.time()

    print("Nile River Data")
    print("---------------")
    print(res)
    print("Time taken: {:.2f} seconds".format(end_time - start_time))

    ##################################################
    # Example - Seatbelt Data - Local Level Model

    seatbelt_data = read_seatbelt_data()
    y_timeseries = seatbelt_data.Drivers

    # Parameters [ H, Q_local, Q_seasonal ]
    initial_parameter_values = array([0.01, 0.01, 0.001])
    parameter_bounds = array([(0, inf), (0, inf), (0, inf)])

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
    )
    end_time = time.time()

    print("Road Traffic Accident Data")
    print("--------------------------")
    print(res)
    print("Time taken: {:.2f} seconds".format(end_time - start_time))
