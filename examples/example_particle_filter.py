import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

from sstspack.ParticleFilterClass import ParticleFilter
from sstspack.ParticleModelDesign import get_local_level_particle_model_design
from sstspack.GaussianModelDesign import get_local_level_model_design
from sstspack import DynamicLinearGaussianModel as DLGM

from example_nile_data import read_nile_data
import plot_figs


if __name__ == "__main__":
    y_timeseries = read_nile_data()
    H = 15099
    Q = 1469.1
    a1 = 0
    P1 = 10 ** 7

    model_design = get_local_level_particle_model_design(y_timeseries, H, Q)
    pfilter = ParticleFilter(y_timeseries, model_design, a1, P1)
    pfilter.filter()

    gaussian_model_design = get_local_level_model_design(y_timeseries.index, Q, H)
    gaussian_model = DLGM(y_timeseries, gaussian_model_design, a1, P1)
    gaussian_model.filter()

    print("plotting")
    plot_figs.plot_fig121(pfilter, gaussian_model)
    print("finished")
