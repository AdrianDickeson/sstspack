import pandas as pd
import matplotlib.pyplot as plt
from numpy import log

import plot_figs


def read_uk_visits_abroad_data():
    """"""
    data_df = pd.read_csv("data/ott.csv")
    index_col = data_df.columns[0]
    data_col = data_df.columns[-3]

    data_series = data_df[data_col]
    data_series.index = data_df[index_col]

    start_date = "1980 JAN"
    end_date = "2006 DEC"
    start_index = data_series.index.tolist().index(start_date)
    end_index = data_series.index.tolist().index(end_date) + 1
    data_series = data_series[data_series.index[start_index:end_index]]

    for idx in data_series.index:
        data_series[idx] = float(data_series[idx])

    return data_series


if __name__ == "__main__":
    y_series = read_uk_visits_abroad_data()
    ylog_series = y_series.copy()
    for idx in ylog_series.index:
        ylog_series[idx] = log(ylog_series[idx])

    plot_figs.plot_fig141(y_series, ylog_series)

    plt.show()
