from numpy.random import normal, choice
from numpy import zeros, sqrt, array, exp, log, pi as PI
import pandas as pd


class ParticleFilter(object):
    """"""

    def __init__(self, y_series, model_design_df, a1, P1, resample_level):
        """"""
        self.N = 10000
        self.resample_level = resample_level
        estimation_columns = ["a", "P"]
        self.model_design = model_design_df.copy()
        self.model_design.insert(0, "y", y_series)
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )

        estimation_columns = [
            "ESS",
            "particles_prior",
            "particles_posterior",
            "weights_prior",
            "weights_posterior",
            "a_prior",
            "a_posterior",
            "P_prior",
            "P_posterior",
        ]
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )
        self.model_design[estimation_columns] = pd.NA

        self.a_prior[self.index_initial] = a1
        self.P_prior[self.index_initial] = P1

    def filter(self):
        """"""
        self.weights_prior[self.index_initial] = zeros(self.N)

        initial_weights = array([1 / self.N] * self.N)

        for idx, key in enumerate(self.index):
            # Step 1: Sample particles from importance distribution
            if idx == 0:
                self.particles_prior[key] = normal(
                    loc=self.a_prior[key],
                    scale=sqrt(self.P_prior[key]),
                    size=self.N,
                )
            else:
                prev_key = self.index[idx - 1]
                self.particles_prior[key] = normal(
                    loc=self.particles_posterior[prev_key],
                    scale=sqrt(self.Q[key]),
                    size=self.N,
                )

            # Step 2: Calculate weight for particles
            if idx == 0:
                prev_weights = initial_weights
            else:
                prev_weights = self.weights_posterior[self.index[idx - 1]]

            sigma2 = self.H[key] + self.P_prior[key]

            self.weights_prior[key] = prev_weights * array(
                [
                    exp(
                        -0.5 * log(2 * PI)
                        - 0.5 * log(sigma2)
                        - 0.5 * (self.y[key] - x) ** 2 / sigma2
                    )
                    for x in self.particles_prior[key]
                ]
            )
            self.weights_prior[key] = self.weights_prior[key] / sum(
                self.weights_prior[key]
            )

            # Step 3: Estimate quantities of interest
            for field in self.X[key]:
                full_field = "{}_posterior".format(field)
                self.model_design.loc[key, full_field] = self.X[key][field](
                    self.particles_prior[key], self.weights_prior[key]
                )

            if idx < len(self.index) - 1:
                nxt_key = self.index[idx + 1]
                self.a_prior[nxt_key] = self.a_posterior[key]
                self.P_prior[nxt_key] = self.P_posterior[key]

            self.ESS[key] = self.effective_sample_size(key)

            # Step 4: Calculate posterior particles
            if self.ESS[key] < self.resample_level:
                idx_resample = choice(
                    range(self.N), self.N, replace=True, p=self.weights_prior[key]
                )
                self.particles_posterior[key] = self.particles_prior[key][idx_resample]
                self.weights_posterior[key] = self.weights_prior[key][idx_resample]
            else:
                self.particles_posterior[key] = self.particles_prior[key].copy()
                self.weights_posterior[key] = self.weights_prior[key].copy()

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
        result = sum(self.weights_prior[key] ** 2) ** -1
        return result
