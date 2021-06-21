import re
import time

import pandas as pd
from numpy import array, diag, full, zeros, identity, ones, inf
import matplotlib.pyplot as plt

import plot_figs
from sstspack import DynamicLinearGaussianModel as DLGM, fitting as fit
import sstspack.GaussianModelDesign as md
from sstspack.Utilities import block_diag


def read_gas_data():
    """"""
    data = []
    fn = "data/gas.dat"
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                elements = elements[1]
                elements = full((1, 1), float(elements))
                data.append(elements)
    return pd.Series(data)


def gas_consumption_model(parameters, model_template, y_timeseries, dt):
    Q_trend = diag(parameters[:2])
    H_trend = full((1, 1), parameters[2])
    Q_seasonal_list = parameters[3:]
    H_seasonal = zeros((1, 1))

    model1 = md.get_local_linear_trend_model_design(y_timeseries, Q_trend, H_trend)
    model2 = md.get_frequency_domain_seasonal_model_design(
        y_timeseries, 4, Q_seasonal_list, H_seasonal
    )

    return md.combine_model_design([model1, model2])


if __name__ == "__main__":
    y_timeseries = read_gas_data()

    initial_parameter_values = array(
        [9.28044e-11, 6.85512e-06, 0.00231959, 0.000927213, 0.000378523]
    )  # ones(5)
    parameter_bounds = array([(0, inf)] * 5)
    parameter_names = array(["H", "LT1", "LT2", "S1", "S2"])

    gas_model_function = gas_consumption_model
    gas_model_template = gas_consumption_model(
        initial_parameter_values, None, y_timeseries, None
    )

    a0 = zeros((5, 1))
    P0 = identity(5)
    diffuse_states = full(5, True)

    # start_time = time.time()
    # res = fit.fit_model_max_likelihood(
    #     initial_parameter_values,
    #     parameter_bounds,
    #     gas_model_function,
    #     y_timeseries,
    #     a0,
    #     P0,
    #     diffuse_states,
    #     gas_model_template,
    #     parameter_names,
    # )
    # end_time = time.time()

    # model = res.model
    model = DLGM(
        y_timeseries, gas_model_template, a0, P0, diffuse_states=diffuse_states
    )
    model.disturbance_smoother()

    # print("Gas Consumption Data")
    # print("--------------------")
    # print(res)
    # print("Time taken: {:.2f} seconds\n".format(end_time - start_time))

    plot_figs.plot_fig144(model, None)

    plt.show()
    print("finished")
