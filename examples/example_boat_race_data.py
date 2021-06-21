import re

import pandas as pd
from numpy import array
import matplotlib.pyplot as plt


def read_boat_race_data():
    """"""
    data = []
    fn = "data/boat.dat"
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                elements = elements[1]
                elements = float(elements)
                data.append(elements)
    return pd.Series(data)


if __name__ == "__main__":
    y_timeseries = read_boat_race_data()
    missing_data = y_timeseries == -9999.99
    y_timeseries[missing_data] = 0.5

    fig, ax = plt.subplots(1)
    ax.plot(y_timeseries)
    plt.show()
