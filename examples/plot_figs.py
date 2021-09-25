import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt, linspace, array, hstack, dot, ravel, exp
from scipy.stats import norm, gaussian_kde
import matplotlib.ticker as mticker

from sstspack import DynamicLinearGaussianModel as DLGM
from plot_tools import (
    get_fig_data,
    plot_state,
    plot_line,
    plot_scatter_line,
    plot_line_and_scatter,
    plot_qq,
    plot_correlogram,
    auto_covariance,
    auto_correlation,
    plot_histogram,
)

NILE_DATA_TITLE = "Volume of Nile river at Aswan 1871-1970"
SEATBELT_DATA_TITLE = "Great Britain Road Accident Casualties 1969-1984"
BOX_JENKINS_DATA_TITLE = "Box-Jenkins modeling of internet user data"
MOTORCYCLE_DATA_TITLE = "Simulated Motorcycle Acceleration Data"
VAN_DATA_TITLE = "Van Road Accident Fatalities 1969-1984"
GAS_DATA_TITLE = "UK Gas Consumption 1960-1986"
EXCHANGE_RATE_DATA_TITLE = "USD/GBP Daily Returns"
UK_VISITORS_DATA_TITLE = "UK Visitors Abroad 1980-2006"

XLABEL = "year"
FIG_LAYOUT = [0, 0.03, 1, 0.95]


def plot_fig21(ssmodel):
    """"""
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.1".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Filtered mean")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, "P_prior (P_t)", (5000, 17500), "variance")

    ax = axs[1, 0]
    data_series = ssmodel.y - ssmodel.a_prior
    plot_scatter_line(ax, data_series, "Forecast error", (-450, 450), "error")

    ax = axs[1, 1]
    plot_line(ax, ssmodel.F, "F (F_t)", (20000, 32500), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.1.pdf")


def plot_fig22(ssmodel):
    """"""
    state_data_df = get_fig_data(ssmodel, "a_hat", "V")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.2".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Smoothed mean")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.V, "V (V_t)", (2200, 4100), "variance")

    ax = axs[1, 0]
    plot_scatter_line(
        ax, ssmodel.r, "Smoothing cumulant r (r_t)", (-0.04, 0.024), "error", XLABEL
    )

    ax = axs[1, 1]
    plot_line(ax, ssmodel.N, "N (N_t)", (0.000048, 0.00011), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.2.pdf")


def plot_fig23(ssmodel):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.3".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(ax, ssmodel.epsilon_hat, "epsilon_hat", (-380, 300), "error")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.epsilon_hat_sigma2, "epsilon_hat_sigma2", None, "variance")

    ax = axs[1, 0]
    plot_scatter_line(ax, ssmodel.eta_hat, "eta_hat", (-55, 35), "error", XLABEL)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.eta_hat_sigma2, "eta_hat_sigma2", None, "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.3.pdf")


def plot_fig24(ssmodel, sim):
    """"""
    initial_key = sim.index[0]
    a0 = ssmodel.a_hat[initial_key]
    P0 = ssmodel.V[initial_key]
    model_fields = ["Z", "d", "H", "T", "c", "R", "Q"]
    model_df = ssmodel.model_data_df[model_fields].copy()
    new_sim = DLGM.simulate_model(model_df, a0, P0)
    plot_ssm = DLGM(new_sim.y, model_df, a0, P0)
    plot_ssm.smoother()
    plot_sim = sim.alpha - ssmodel.a_hat + plot_ssm.a_hat

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.4".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_line_and_scatter(
        ax, ssmodel.a_hat, plot_sim, "a_hat simulation", None, "volume"
    )

    ax = axs[0, 1]
    plot_line_and_scatter(
        ax, ssmodel.a_hat, sim.alpha, "a_hat conditional simulation", None, "volume"
    )

    ax = axs[1, 0]
    plot_line_and_scatter(
        ax,
        ssmodel.epsilon_hat,
        sim.epsilon,
        "epsilon_hat conditional simulation",
        None,
        "error",
        XLABEL,
    )

    ax = axs[1, 1]
    plot_line_and_scatter(
        ax,
        ssmodel.eta_hat[:-1],
        sim.eta[:-1],
        "eta_hat conditional simulation",
        None,
        "error",
        XLABEL,
    )

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.4.pdf")


def plot_fig25(ssmodel):
    """"""
    missing_mask = ssmodel.y.apply(lambda x: not DLGM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.5".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Filtered mean", missing_mask)

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, "P_prior (P_t)", (5000, 36000), "variance")

    state_data_df = get_fig_data(ssmodel, "a_hat", "V")

    ax = axs[1, 0]
    plot_state(ax, ssmodel, state_data_df, "Smoothed mean", missing_mask)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.V, "V (V_t)", (2200, 10000), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.5.pdf")


def plot_fig26(ssmodel):
    """"""
    missing_mask = ssmodel.y.apply(lambda x: not DLGM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior", confidence=0.5)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.6".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(
        ax,
        ssmodel,
        state_data_df,
        "Filtered mean",
        missing_mask,
        xlim=(1868, 2003),
        confidence=0.5,
    )

    ax = axs[0, 1]
    plot_line(
        ax,
        ssmodel.P_prior,
        "P_prior (P_t)",
        (5000, 50000),
        "variance",
        xlim=(1868, 2003),
    )

    ax = axs[1, 0]
    plot_line(
        ax,
        ssmodel.a_prior,
        "a_prior (a_t)",
        (700, 1250),
        "volume",
        XLABEL,
        xlim=(1868, 2003),
    )

    ax = axs[1, 1]
    plot_line(
        ax, ssmodel.F, "F (F_t)", (20000, 65000), "variance", XLABEL, xlim=(1868, 2003)
    )

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.6.pdf")


def plot_fig27(ssmodel):
    """"""
    data_series = ssmodel.model_data_df.apply(
        lambda x: x["v"].ravel()[0] / sqrt(x["F"].ravel()[0]), axis=1
    )

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.7".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - epsilon", (-3.5, 3), "error"
    )

    ax = axs[0, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    ax = axs[1, 0]
    plot_qq(ax, data_series, "QQ plot", (-3.5, 3))

    ax = axs[1, 1]
    mean = data_series.sum() / len(data_series)
    acf = [auto_correlation(data_series - mean, i) for i in range(1, 11)]
    variance = 1 / len(data_series)
    plot_correlogram(ax, acf, variance, "Correlogram")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.7.pdf")


def plot_fig28(ssmodel):
    """"""
    data_series = ssmodel.model_data_df.apply(
        lambda x: x["u"].ravel()[0] / sqrt(x["D"].ravel()[0]), axis=1
    )

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.8".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - epsilon", (-3.5, 3), "error"
    )

    ax = axs[0, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    data_series = ssmodel.model_data_df.apply(
        lambda x: x["r"].ravel()[0] / sqrt(x["N"].ravel()[0]), axis=1
    )

    ax = axs[1, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - eta", (-3.5, 3), "error"
    )

    ax = axs[1, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.8.pdf")


def run_diagnostics(ssmodel):
    """"""
    forecast_errors = ssmodel.model_data_df.apply(
        lambda x: x["v"].ravel()[0] / sqrt(x["F"].ravel()[0]), axis=1
    )
    n = len(forecast_errors)

    m_1 = forecast_errors.sum() / n
    forecast_errors = forecast_errors - m_1

    m_dict = {i: forecast_errors.apply(lambda x: x ** i).sum() / n for i in [2, 3, 4]}
    S = m_dict[3] / sqrt(m_dict[2] ** 3)
    K = m_dict[4] / m_dict[2] ** 2 - 3
    N = n * (S ** 2 / 6 + K ** 2 / 24)

    h = 33
    h_series = forecast_errors.apply(lambda x: x ** 2)
    H = h_series.iloc[:h].sum() / h_series.iloc[h:].sum()

    q = 9
    q_dict = {i: auto_correlation(forecast_errors, i) for i in range(1, q + 1)}
    Q = n * (n + 2) * sum(q_dict[i] ** 2 / (n - i) for i in range(1, q + 1))

    print("Diagnostic checks")
    print("=================\n")
    print(
        "S: {:.2f}, K: {:.2f}, N: {:.2f}, H({}): {:.2f}, Q({}): {:.2f}".format(
            S, K, N, h, H, q, Q
        )
    )


def plot_fig81(seatbelt_df):
    """"""
    fig, ax = plt.subplots()
    fig.suptitle("{} - Fig. 8.1".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.plot(seatbelt_df.index, seatbelt_df.iloc[:, 0], "-bx")
    ax.set_ylabel("Log KSI")
    ax.set_xlabel("Year")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.1.pdf")


def plot_fig82(model):
    """"""
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.2".format(SEATBELT_DATA_TITLE), fontsize=14)

    plot_seasonal_breakdown(axs, model, [0], range(1, 12))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.2.pdf")


def plot_seasonal_breakdown(axs, model, level_indexes, seasonal_indexes):
    """"""
    level_data = array(
        [
            dot(model.Z[idx][:, level_indexes], model.a_hat[idx][level_indexes, :])
            for idx in model.index
        ]
    ).T
    seasonal_data = array(
        [
            dot(
                model.Z[idx][:, seasonal_indexes], model.a_hat[idx][seasonal_indexes, :]
            )
            for idx in model.index
        ]
    ).T
    ax = axs[0]
    ax.plot(model.index, model.y, "g-.")
    ax.plot(model.index, ravel(level_data), "r-")
    ax.set_title("Level")

    ax = axs[1]
    ax.plot(model.index, ravel(seasonal_data))
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_title("Seasonality")

    ax = axs[2]
    ax.plot(model.index, model.epsilon_hat)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_title("Irregular")


def plot_fig83(model):
    a_hat = hstack(model.a_hat)
    a_prior = hstack(model.a_prior)
    fig, ax = plt.subplots(1)
    fig.suptitle("{} - Fig. 8.3".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.scatter(x=model.index, y=model.y, marker=".", label="Data")
    ax.plot(model.index, a_hat[0, :], "r--", label="Smoothed")
    ax.plot(model.index[12:], a_prior[0, 12:], "g--", label="Predicted")
    ax.set_ylim((6.9, 7.9))
    ax.legend()

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.3.pdf")


def plot_lines(data_series, ax):
    """"""
    for idx in data_series.index:
        try:
            value = data_series[idx][0, 0]
        except:
            value = data_series[idx]
        ax.plot((idx, idx), (0, value), c="k")


def plot_fig84(model, confidence=0.9):
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.4".format(SEATBELT_DATA_TITLE), fontsize=14)

    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    nondiffuse_index = model.index[12:]
    prediction_residuals = model.v[nondiffuse_index]
    residuals_std = ravel(sqrt(model.F[nondiffuse_index].to_list()))
    standardise_residuals = prediction_residuals / residuals_std
    auxillary_residuals_u = model.u[nondiffuse_index] / ravel(
        sqrt(model.D[nondiffuse_index].to_list())
    )
    level_idx = [idx for idx in model.index if model.r[idx] is not pd.NA]
    level_data = [model.r[idx][0, 0] for idx in level_idx]
    level_series = pd.Series(level_data, index=level_idx)
    level_std = array([sqrt(model.N[idx][0, 0]) for idx in level_idx])
    aux_level_resid = level_series / level_std

    ax = axs[0]
    ax.plot(standardise_residuals.index, standardise_residuals)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    ax = axs[1]
    plot_lines(auxillary_residuals_u, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    ax = axs[2]
    plot_lines(aux_level_resid, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.4.pdf")


def plot_fig85(model):
    """"""
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.5".format(SEATBELT_DATA_TITLE), fontsize=14)

    plot_seasonal_breakdown(axs, model, [0, 12, 13], range(1, 12))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.5.pdf")


def plot_fig86(data_df):
    """"""
    fig, ax = plt.subplots(1)
    fig.suptitle("{} - Fig. 8.6".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.plot(data_df.index, data_df.Front, "-.r", label="Front Seat")
    ax.plot(data_df.index, data_df.Rear, "--xg", label="Rear Seat")
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Log KSI")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.6.pdf")


def plot_fig87(model):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.7".format(SEATBELT_DATA_TITLE), fontsize=14)

    confidence = 0.95
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)

    data = hstack([model.y[idx] for idx in model.index])
    seasonal = [False] * 2 + [True] * 22 + [False] * 6
    front_level = []
    front_sd = []
    rear_level = []
    rear_sd = []
    for idx in model.index:
        Z = model.Z[idx].copy()
        Z[:, seasonal] = 0
        var_mat = dot(dot(Z, model.V[idx]), Z.T)
        front_level.append(dot(Z[0, :], model.a_hat[idx]))
        front_sd.append(sqrt(var_mat[0, 0]))
        rear_level.append(dot(Z[1, :], model.a_hat[idx]))
        rear_sd.append(sqrt(var_mat[1, 1]))

    front_level = ravel(array(front_level))
    front_sd = ravel(array(front_sd))
    front_upper = ravel(front_level + quantile * front_sd)
    front_lower = ravel(front_level - quantile * front_sd)

    rear_level = ravel(array(rear_level))
    rear_sd = ravel(array(rear_sd))
    rear_upper = ravel(rear_level + quantile * rear_sd)
    rear_lower = ravel(rear_level - quantile * rear_sd)

    ax = axs[0]
    ax.plot(model.index, data[0, :], "--xb", label="Front Seat")
    ax.plot(model.index, ravel(front_level), "-r", label="Level")
    ax.fill_between(model.index, front_upper, front_lower, alpha=0.5, label="95% Conf.")
    ax.legend()

    ax = axs[1]
    ax.plot(model.index, data[1, :], "--xb", label="Rear Seat")
    ax.plot(model.index, ravel(rear_level), "-r", label="Level")
    ax.fill_between(model.index, rear_upper, rear_lower, alpha=0.5, label="95% Conf.")
    ax.legend()

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.7.pdf")


def plot_with_missing_data(data, ax):
    """"""
    start_idx = None
    data_idx_list = []
    for curr_idx in range(1, len(data) + 1):
        if start_idx is None and data[curr_idx] is not pd.NA:
            start_idx = curr_idx
        if data[curr_idx] is pd.NA and start_idx is not None:
            data_idx_list.append(list(range(start_idx, curr_idx)))
            start_idx = None
    for plot_range in data_idx_list:
        ax.plot(plot_range, data[plot_range], "-b")


def plot_fig88(data, missing_data):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.8".format(BOX_JENKINS_DATA_TITLE), fontsize=14)

    ax = axs[0]
    ax.plot(data)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_ylabel("Change")

    ax = axs[1]
    plot_with_missing_data(missing_data, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_ylabel("Change")
    ax.set_xlabel("Minutes")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.8.pdf")


def plot_fig89(model):
    """"""
    fig, ax = plt.subplots(1)
    ax.scatter(
        model.index[:99], model.y[model.index[:99]], label="Change in internet users"
    )
    forecast_data = pd.Series(
        [dot(model.Z[idx], model.a_prior[idx])[0, 0] for idx in model.index],
        index=model.index,
    )

    confidence = 0.5
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    forecast_error = pd.Series(
        [quantile * sqrt(x[0, 0]) for x in model.F], index=model.index
    )

    ax.plot(
        forecast_data.index[(model.d_diffuse + 1) : 100],
        forecast_data[(model.d_diffuse + 1) : 100],
        "k",
        label="One step ahead forecast",
    )
    ax.plot(forecast_data.index[100:], forecast_data[100:], "--k")

    ax.plot(
        forecast_error.index[99:],
        forecast_error[forecast_error.index[99:]],
        "g",
        label="50% Confidence interval",
    )
    ax.plot(forecast_error.index[99:], -forecast_error[forecast_error.index[99:]], "g")

    ax.legend()
    ax.set_title("{} ARMA(1, 1) - Fig. 8.9".format(BOX_JENKINS_DATA_TITLE))
    ax.set_ylabel("Change")
    ax.set_xlabel("Minutes")

    fig.savefig("figures/fig8.9.pdf")


def plot_fig810(model):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.10".format(MOTORCYCLE_DATA_TITLE), fontsize=14)

    ax = axs[0]
    data_index = []
    data_full = []
    epsilon_full = []
    for idx in model.index:
        for i in range(model.y[idx].shape[0]):
            data_index.append(idx)
            data_full.append(model.y[idx][i, 0])
            epsilon_full.append(model.epsilon_hat[idx][i, 0] / sqrt(model.H[idx][i, i]))
    a_hat = array([model.a_hat[idx][0, 0] for idx in model.index])
    confidence = 0.95
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    V_sqrt95 = quantile * array([sqrt(model.V[idx][0, 0]) for idx in model.index])

    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.scatter(data_index, data_full, s=5, c="b", marker="x", label="Simulated data")
    ax.plot(model.index, a_hat, "g-", label="Spline model")
    ax.plot(model.index, a_hat + V_sqrt95, "r--", label="95% Confidence")
    ax.plot(model.index, a_hat - V_sqrt95, "r--")
    ax.legend()
    ax.set_ylabel("Accelaration")
    ax.set_xlabel("Time")

    ax = axs[1]
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.scatter(data_index, epsilon_full, s=5, c="b")
    ax.set_ylabel("Standardise residual")
    ax.set_xlabel("Time")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.10.pdf")


def figs12(axs, particle_model, gaussian_model):
    """"""
    ax = axs[0, 0]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(particle_model.index, particle_model.y, marker="x", label="Data")
    confidence90 = norm.ppf(0.95) * array([sqrt(x) for x in particle_model.P_posterior])
    ax.plot(
        particle_model.index,
        particle_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(particle_model.index, particle_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[0, 1]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "--", label="Particle Filter"
    )
    ax.plot(gaussian_model.index, gaussian_model.a_posterior, label="Kalman Filter")
    ax.legend()
    ax.set_ylim((700, 1200))

    ax = axs[1, 0]
    ax.plot(
        particle_model.index, particle_model.P_posterior, "--", label="Particle Filter"
    )
    ax.plot(gaussian_model.index, gaussian_model.P_posterior, label="Kalman Filter")
    ax.legend()
    ax.set_ylim((0, 15000))

    ax = axs[1, 1]
    ax.plot(particle_model.index, particle_model.ESS)
    ax.set_ylim((0, 10000))


def plot_fig121(particle_model, gaussian_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.1".format(NILE_DATA_TITLE), fontsize=14)

    figs12(axs, particle_model, gaussian_model)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.1.pdf")


def plot_fig122(particle_model, gaussian_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.2".format(NILE_DATA_TITLE), fontsize=14)

    figs12(axs, particle_model, gaussian_model)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.2.pdf")


def plot_fig123(particle_model, auxiliary_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.3".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(particle_model.index, particle_model.y, marker="x", label="Data")
    confidence90 = norm.ppf(0.95) * array([sqrt(x) for x in particle_model.P_posterior])
    ax.plot(
        particle_model.index,
        particle_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(particle_model.index, particle_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[0, 1]
    ax.plot(particle_model.index, particle_model.ESS)
    ax.set_ylim((0, 10000))

    ax = axs[1, 0]
    ax.plot(
        auxiliary_model.index, auxiliary_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(auxiliary_model.index, auxiliary_model.y, marker="x", label="Data")
    confidence90 = norm.ppf(0.95) * array(
        [sqrt(x) for x in auxiliary_model.P_posterior]
    )
    ax.plot(
        auxiliary_model.index,
        auxiliary_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(auxiliary_model.index, auxiliary_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[1, 1]
    ax.plot(auxiliary_model.index, auxiliary_model.ESS)
    ax.set_ylim((0, 10000))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.3.pdf")


def plot_fig141(y_series, ylog_series):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 14.1".format(UK_VISITORS_DATA_TITLE), fontsize=14)

    ax = axs[0]
    ax.plot(y_series.index, y_series)
    ax.set_ylabel("Visitors (thousands)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    ax = axs[1]
    ax.plot(ylog_series.index, ylog_series)
    ax.set_ylabel("Log Visitors (thousands)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig14.4.pdf")


def plot_fig142(extended_model, c_0, c_mu):
    """"""
    a_hat = hstack([extended_model.Z[idx][0, 0] for idx in extended_model.index])
    residuals = hstack(
        [
            extended_model.y[idx] - extended_model.a_hat[idx][0, 0]
            for idx in extended_model.index
        ]
    )

    fig, axs = plt.subplots(2, 1)
    fig.suptitle(f"{UK_VISITORS_DATA_TITLE} - Fig. 14.2", fontsize=14)

    ax = axs[0]
    # ax.plot(extended_model.index, a_hat[2, :])
    ax.scatter(extended_model.index, residuals)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    ax = axs[1]
    # ax.plot(extended_model.index, a_hat)
    # for i in [1]:
    #     ax.plot(
    #         extended_model.index,
    #         hstack([extended_model.a_hat[idx][i, 0] for idx in extended_model.index]),
    #     )
    Z = array(
        [
            extended_model.a_hat[idx][0, 0]
            + exp(c_0 + c_mu * extended_model.a_hat[idx][0, 0])
            * extended_model.a_hat[idx][2, 0]
            for idx in extended_model.index
        ]
    )
    ax.plot(extended_model.index, Z)
    ax.scatter(extended_model.index, extended_model.y, s=2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig14.2.pdf")


def plot_fig143(y_series, model):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 14.3".format(VAN_DATA_TITLE), fontsize=14)

    ax = axs[0]
    ax.plot(y_series.index, y_series)
    ax.scatter(y_series.index, y_series)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig14.3.pdf")


def plot_fig144(gaussian_model, non_gaussian_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 14.4".format(GAS_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    mask = array([0, 0, 1, 1, 1])
    ax.plot(gaussian_model.aggregate_field("a_hat", mask))

    ax = axs[1, 0]
    ax.plot(gaussian_model.epsilon_hat)
    ax.set_ylim((-0.4, 0.4))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig14.4.pdf")


def plot_fig145(y_series, transformed_series, model):
    """"""
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 14.5".format(GAS_DATA_TITLE), fontsize=14)

    ax = axs[0]
    ax.plot(y_series)

    ax = axs[1]
    ax.scatter(x=transformed_series.index, y=transformed_series, s=1)
    ax.set_ylim((-30, 0))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig14.5.pdf")
