import re

import pandas as pd
from numpy import array
import matplotlib.pyplot as plt


def read_motorcycle_data():
    """"""
    data = []
    fn = "data/motorcycle.dat"
    index = []
    with open(fn, "r") as f:
        for idx, line in enumerate(f):
            if idx > 1:
                line = line.replace("\t", " ")
                line = line.replace("\n", " ")
                elements = re.split(" +", line)
                index.append(float(elements[1]))
                elements = elements[2:4]
                elements = [float(val) for val in elements]
                data.append(elements)
    data = array(data)
    data_df = pd.DataFrame(data, index=index)
    data_df.columns = ["Time Interval", "Motorcycle Acceleration"]
    return data_df


if __name__ == "__main__":
    data = read_motorcycle_data()
    print(data)
    plt.plot(data.index, data["Motorcycle Acceleration"])
    plt.show()
