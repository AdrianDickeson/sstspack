import pandas as pd
import matplotlib.pyplot as plt
from numpy import log

import plot_figs


def read_exchange_rate_data():
    """"""
    return pd.read_csv("data/sv.dat") / 100


if __name__ == "__main__":
    y_timeseries = read_exchange_rate_data()

    transformed_series = log(y_timeseries ** 2)

    plot_figs.plot_fig145(y_timeseries, transformed_series, None)

    plt.show()
