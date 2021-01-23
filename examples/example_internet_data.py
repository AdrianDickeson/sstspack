import re

import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

import plot_figs


def get_internet_data():
    """"""
    data = []
    fn = "data/internet.dat"
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                elements = elements[1:3]
                elements = [float(val) for val in elements]
                data.append(elements)
    data = array(data)
    data_df = pd.DataFrame(data)
    data_df.columns = ["Users", "Change"]
    return data_df


if __name__ == "__main__":
    data = get_internet_data()["Change"]
    missing_data = data.copy()
    missing_idx = [5, 15, 25, 35, 45, 55, 65, 71, 72, 73, 74, 75, 85, 95]
    missing_data[missing_idx] = pd.NA

    plot_figs.plot_fig88(data, missing_data)
    plt.show()
