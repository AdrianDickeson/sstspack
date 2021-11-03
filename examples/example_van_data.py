import re
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt

import plot_figs
from example_seatbelt_data import eom


def read_van_data():
    """"""
    data = []
    fn = "data/van.dat"
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                col = 0
                while elements[col] == "":
                    col += 1
                elements = elements[col]
                elements = float(elements)
                data.append(elements)
    index = [
        eom(dt.date(year, month, 1))
        for year in range(1969, 1985)
        for month in range(1, 13)
    ]
    return pd.Series(data, index=index[:-1])


def main():
    y_series = read_van_data()

    plot_figs.plot_fig143(y_series, None)

    plt.show()


if __name__ == "__main__":
    main()
