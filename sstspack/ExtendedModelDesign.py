import pandas as pd
from numpy import zeros, identity

import sstspack.GaussianModelDesign as md


def get_nonlinear_model_design(
    length_index, Z_fn, T_fn, R_fn, Q_fn, H_fn, Z_prime_fn=None, T_prime_fn=None
):
    """"""
    if Z_prime_fn is None and T_prime_fn is None:
        return md.get_static_model_df(
            length_index, Z_fn=Z_fn, H_fn=H_fn, T_fn=T_fn, R_fn=R_fn, Q_fn=Q_fn
        )
    if T_prime_fn is None:
        return md.get_static_model_df(
            length_index,
            Z_fn=Z_fn,
            H_fn=H_fn,
            T_fn=T_fn,
            R_fn=R_fn,
            Q_fn=Q_fn,
            Z_prime_fn=Z_prime_fn,
        )
    if Z_prime_fn is None:
        return md.get_static_model_df(
            length_index,
            Z_fn=Z_fn,
            H_fn=H_fn,
            T_fn=T_fn,
            R_fn=R_fn,
            Q_fn=Q_fn,
            T_prime_fn=T_prime_fn,
        )

    return md.get_static_model_df(
        length_index,
        Z_fn=Z_fn,
        H_fn=H_fn,
        T_fn=T_fn,
        R_fn=R_fn,
        Q_fn=Q_fn,
        Z_prime_fn=Z_prime_fn,
        T_prime_fn=T_prime_fn,
    )
