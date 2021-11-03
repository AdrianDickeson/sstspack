import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

from sstspack.ParticleFilterClass import ParticleFilter
from sstspack.ParticleModelDesign import get_local_level_particle_model_design
from sstspack.GaussianModelDesign import get_local_level_model_design
from sstspack import DynamicLinearGaussianModel as DLGM

from example_nile_data import read_nile_data
import plot_figs


def main():
    y_timeseries = read_nile_data()
    H = 15099
    Q = 1469.1
    a1 = 0
    P1 = 10 ** 7

    print("Filtering Nile data... ")
    print("Kalman filter... ", end="")
    gaussian_model_design = get_local_level_model_design(y_timeseries.index, Q, H)
    gaussian_model = DLGM(y_timeseries, gaussian_model_design, a1, P1)
    gaussian_model.filter()
    print("finished")

    print("Particle filter... ", end="")
    N = 10000
    model_design = get_local_level_particle_model_design(y_timeseries, H, Q)
    pfilter = ParticleFilter(y_timeseries, model_design, a1, P1, N, 1)
    pfilter.filter()
    print("finished")

    print("Particle filter with resampling... ", end="")
    pfilter_resampling = ParticleFilter(y_timeseries, model_design, a1, P1, N, 10000)
    pfilter_resampling.filter()
    print("finished")

    print("Auxiliary particle filter... ", end="")
    pfilter_aux = ParticleFilter(y_timeseries, model_design, a1, P1, N, 10000, aux=True)
    pfilter_aux.filter()
    print("finished")

    print("Plotting figures... ", end="")
    plot_figs.plot_fig121(pfilter, gaussian_model)
    plot_figs.plot_fig122(pfilter_resampling, gaussian_model)
    plot_figs.plot_fig123(pfilter_resampling, pfilter_aux)
    print("finished")


if __name__ == "__main__":
    main()
