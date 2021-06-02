import pandas as pd

import sstspack.GaussianModelDesign as md


def identity(x):
    """"""
    return x


def get_local_level_particle_model_design(length_index, H, Q):
    """"""

    def a(particles, weights):
        return sum(weights * particles)

    def P(particles, weights):
        return sum(weights * particles ** 2) - a(particles, weights) ** 2

    X = {"a": a, "P": P}

    H = H
    Q = Q

    result = md.get_static_model_df(length_index, H=H, Q=Q, X=X)

    return result
