import pandas as pd


def identity(x):
    """"""
    return x


def get_local_level_particle_model_design(y_timeseries, H, Q):
    """"""

    def a(particles, weights):
        return sum(weights * particles)

    def P(particles, weights):
        return sum(weights * particles ** 2) - a(particles, weights) ** 2

    X = {"a": a, "P": P}

    y_length = len(y_timeseries)
    data = {
        "H": [H] * y_length,
        "Q": [Q] * y_length,
        "a_prior": [pd.NA] * y_length,
        "P_prior": [pd.NA] * y_length,
        "a_posterior": [pd.NA] * y_length,
        "P_posterior": [pd.NA] * y_length,
        "X": [X] * y_length,
    }
    result = pd.DataFrame(data, index=y_timeseries.index)
    return result