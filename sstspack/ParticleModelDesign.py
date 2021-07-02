import pandas as pd
import numpy as np

import sstspack.GaussianModelDesign as md
from sstspack.Utilities import identity_fn


def identity_matrix(x):
    """"""
    return np.identity(x.shape[0])


def get_local_level_particle_model_design(length_index, H, Q):
    """"""

    def a(particles, weights):
        return sum(weights * particles)

    def P(particles, weights):
        return sum(weights * particles ** 2) - a(particles, weights) ** 2

    X = {"a": a, "P": P}

    H = H
    Q = Q

    Z_fn = identity_fn
    T_fn = identity_fn
    R_fn = identity_matrix

    result = md.get_static_model_df(
        length_index, H=H, Q=Q, X=X, Z_fn=Z_fn, T_fn=T_fn, R_fn=R_fn
    )

    return result
