import pandas as pd
from numpy import zeros, identity

import sstspack.GaussianModelDesign as md


def get_nonlinear_model_design(length_index, Z_fn, T_fn, R_fn, Q_fn, H_fn):
    """"""
    return md.get_static_model_df(
        length_index, Z_fn=Z_fn, H_fn=H_fn, T_fn=T_fn, R_fn=R_fn, Q_fn=Q_fn
    )
