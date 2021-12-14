import re
import logging
import time

import pandas as pd
from numpy import array, zeros, identity, vstack, inf, full
import matplotlib.pyplot as plt

from sstspack import (
    GaussianModelDesign as md,
    DynamicLinearGaussianModel as DLGM,
    Fitting as fit,
    Utilities as utl,
)
import plot_figs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def motorcycle_spline_model(parameters, *args, **kwargs):
    y_timeseries = kwargs["y_series"]
    diffuse_states = kwargs["diffuse_states"]
    dt = kwargs["dt"]
    model_design = md.get_spline_smoothing_model_design(
        y_timeseries, parameters[0], parameters[1], dt
    )
    return DLGM(
        y_timeseries, model_design, diffuse_states=diffuse_states, validate_input=False
    )


def main():
    logger.debug("Reading motorcycle data")
    data = read_motorcycle_data()
    y_timeseries = data["Motorcycle Acceleration"]
    dt = array(data["Time Interval"])

    # Parameters [ H, Q ]
    initial_parameter_values = array([10000, 1000])
    parameter_bounds = array([(0, inf), (0, inf)])
    parameter_names = array(["Lambda", "H"])

    motorcycle_model_function = motorcycle_spline_model

    diffuse_states = [True, True]

    logger.debug("Fitting spline model using maximum likelihood")
    start_time = time.time()
    res = fit.fit_model_max_likelihood(
        initial_parameter_values,
        parameter_bounds,
        motorcycle_model_function,
        y_timeseries,
        diffuse_states=diffuse_states,
        dt=dt,
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
    model.disturbance_smoother()

    logger.info("Producing figures")
    plot_figs.plot_fig810(model)


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.DEBUG)

    logger.addHandler(stream_handler)
    main()
    plt.show()
