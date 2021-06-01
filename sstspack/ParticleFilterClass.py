from numpy.random import normal
from numpy import zeros, sqrt, array, exp, log, pi as PI
import pandas as pd


class ParticleFilter(object):
    """"""

    def __init__(self, y_series, model_design_df, a1, P1):
        """"""
        self.N = 10000
        estimation_columns = ["a", "P"]
        self.model_design = model_design_df
        self.model_design.insert(0, "y", y_series)
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )

        estimation_columns = ["ESS", "particles", "weights"]
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )
        self.model_design[estimation_columns] = pd.NA

        self.a_prior[self.index_initial] = a1
        self.P_prior[self.index_initial] = P1

    def filter(self):
        """"""
        self.weights[self.index_initial] = zeros(self.N)

        initial_weights = array([1 / self.N] * self.N)

        for idx, key in enumerate(self.index):
            if idx == 0:
                self.particles[key] = normal(
                    loc=self.a_prior[key],
                    scale=sqrt(self.P_prior[key]),
                    size=self.N,
                )
            else:
                prev_key = self.index[idx - 1]
                self.particles[key] = normal(
                    loc=self.particles[prev_key],
                    scale=sqrt(self.Q[key]),
                    size=self.N,
                )
                # self.particles[key] = self.particles[self.index_initial].copy()

            if idx == 0:
                prev_weights = initial_weights
            else:
                prev_weights = self.weights[self.index[idx - 1]]

            self.weights[key] = prev_weights * array(
                [
                    exp(
                        -0.5 * log(2 * PI)
                        - 0.5 * log(self.H[key])
                        - 0.5 * (self.y[key] - x) ** 2 / self.H[key]
                    )
                    for x in self.particles[key]
                ]
            )
            self.weights[key] = self.weights[key] / sum(self.weights[key])

            for field in self.X[key]:
                full_field = "{}_posterior".format(field)
                self.model_design.loc[key, full_field] = self.X[key][field](
                    self.particles[key], self.weights[key]
                )

            if idx < len(self.index) - 1:
                nxt_key = self.index[idx + 1]
                self.a_prior[nxt_key] = self.a_posterior[key]
                self.P_prior[nxt_key] = self.P_posterior[key]

            self.ESS[key] = self.effective_sample_size(key)

    def __getattr__(self, name):
        """"""
        try:
            return self.model_design[name]
        except KeyError:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

    @property
    def index(self):
        """"""
        return self.model_design.index

    @property
    def index_initial(self):
        """"""
        return self.index[0]

    def effective_sample_size(self, key):
        """"""
        result = sum(self.weights[key] ** 2) ** -1
        return result
