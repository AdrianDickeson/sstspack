import logging

import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

from sstspack.ParticleFilterClass import ParticleFilter
from sstspack.ParticleModelDesign import get_local_level_particle_model_design
from sstspack.GaussianModelDesign import get_local_level_model_design
from sstspack import DynamicLinearGaussianModel as DLGM, Utilities as utl

from example_nile_data import read_y_timeseries
import plot_figs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    y_timeseries = read_y_timeseries()
    H = 15099
    Q = 1469.1
    a_initial = 0
    P_initial = 10 ** 7

    logger.debug("Filtering Nile time series")
    gaussian_model_design = get_local_level_model_design(y_timeseries.index, Q, H)
    gaussian_model = DLGM(y_timeseries, gaussian_model_design, a_initial, P_initial)
    gaussian_model.filter()

    logger.debug("Running basic particle filter")
    N = 10000
    model_design = get_local_level_particle_model_design(y_timeseries, H, Q)
    pfilter = ParticleFilter(y_timeseries, model_design, a_initial, P_initial, N, 1)
    pfilter.filter()

    logger.debug("Running particle filter with resampling")
    pfilter_resampling = ParticleFilter(
        y_timeseries, model_design, a_initial, P_initial, N, 10000
    )
    pfilter_resampling.filter()

    logger.debug("Running auxiliary particle filter")
    pfilter_aux = ParticleFilter(
        y_timeseries, model_design, a_initial, P_initial, N, 10000, aux=True
    )
    pfilter_aux.filter()

    logger.info("Producing figures")
    plot_figs.plot_fig121(pfilter, gaussian_model)
    plot_figs.plot_fig122(pfilter_resampling, gaussian_model)
    plot_figs.plot_fig123(pfilter_resampling, pfilter_aux)


if __name__ == "__main__":
    stream_handler = utl.getSetupStreamHandler(logging.DEBUG)

    logger.addHandler(stream_handler)
    main()
    plt.show()
