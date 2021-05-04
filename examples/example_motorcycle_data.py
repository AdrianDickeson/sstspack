import re

import pandas as pd
from numpy import array, zeros, identity, vstack, inf, full
import matplotlib.pyplot as plt

from sstspack import (
    modeldesign as md,
    DynamicLinearGaussianModel as DLGM,
    fitting as fit,
)
import plot_figs


def read_motorcycle_data():
    """"""
    data = []
    fn = "data/motorcycle.dat"
    index = []
    prev_index = -1
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                curr_index = float(elements[1])
                elements = elements[2:4]
                elements = [float(val) for val in elements]
                elements[1] = array([[elements[1]]])
                if curr_index != prev_index:
                    index.append(curr_index)
                    data.append(elements)
                else:
                    data[-1][1] = vstack((data[-1][1], elements[1]))
                prev_index = curr_index
    data = list(data)
    data_df = pd.DataFrame(data, index=index)
    data_df.columns = ["Time Interval", "Motorcycle Acceleration"]
    return data_df


def motorcycle_spline_model(parameters, model_template, y_timeseries, dt):
    result = md.get_spline_smoothing_model_data(
        y_timeseries, parameters[0], parameters[1], dt
    )
    return result


if __name__ == "__main__":
    print("Reading motorcycle data...", end=" ")
    data = read_motorcycle_data()
    y_timeseries = data["Motorcycle Acceleration"]
    dt = array(data["Time Interval"])
    print("finished")

    # Parameters [ H, Q ]
    initial_parameter_values = array([10000, 1000])
    parameter_bounds = array([(0, inf), (0, inf)])
    parameter_names = array(["Lambda", "H"])

    motorcycle_model_function = motorcycle_spline_model
    motorcycle_model_template = None

    # Diffuse initialisation used - a0, P0 are ignored
    a0 = zeros((2, 1))
    P0 = identity(2)
    diffuse_states = [True, True]

    print("Fitting spline model...", end=" ")
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        motorcycle_model_function,
        y_timeseries,
        a0,
        P0,
        diffuse_states,
        motorcycle_model_template,
        parameter_names,
    )
    print("finished")

    model = res.model
    model.disturbance_smoother()

    print("Producing plots...", end=" ")
    plot_figs.plot_fig810(model)
    print("finished")

    plt.show()
