import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sstspack import DynamicLinearGaussianModel as DLGM, GaussianModelDesign as md

pd.set_option("display.max_columns", 100)

if __name__ == "__main__":
    fn_path = "data/noisy_sin_data.csv"
    data = pd.read_csv(fn_path)
    y = data["Observed"]
    alpha = data["Actual"]

    epsilon = y - alpha
    eta = np.array(alpha)[1:] - np.array(alpha)[:-1]
    H = np.std(epsilon) ** 2
    sigma2_eta = np.std(eta) ** 2

    y[20:22] = pd.NA
    y[22:24] = np.nan
    y[24:26] = None
    y[26:28] = [[pd.NA], [pd.NA]]
    y[28:30] = [[np.nan], [np.nan]]
    y[30:32] = [[None], [None]]
    y[32:34] = [[[pd.NA]], [[pd.NA]]]
    y[34:36] = [[[np.nan]], [[np.nan]]]
    y[36:38] = [[[None]], [[None]]]
    non_missing_mask = [not DLGM.is_all_missing(value) for value in y]

    data_df = md.get_local_level_model_design(100, sigma2_eta, H)
    ssm = DLGM(data["Observed"], data_df, np.array([0.0]), np.array([[1.0]]))
    ssm.filter()
    ssm.smoother()
    ssm.disturbance_smoother()
    sim = ssm.simulate_smoother()

    plt.scatter(data.index[non_missing_mask], y[non_missing_mask], marker="x")
    plt.plot(ssm.a_hat)
    plt.plot(sim["alpha"])
    plt.show()

    #     print('L: {}'.format(ssm.log_likelihood()))
    #     print(ssm.model_data_df.head())
    print(ssm.a_prior_final)
    print(ssm.P_prior_final)
