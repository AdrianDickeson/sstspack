'''
Created on 26 Aug 2017

@author: adriandickeson
'''

# import exceptions as ex

from numpy import dot, transpose as tr, zeros, ones, full, log, pi as PI, identity, array, isnan
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

class StateSpaceModel(object):
    '''
    The StateSpaceModel object based on the user provided parameters.

    Parameters
    ----------
    model_parameters_df : pandas.DataFrame
    '''

    def __init__(self, y_series, model_parameters_df, a_prior_initial, P_prior_initial):
        '''
        Constructor
        '''
        self.a0 = a_prior_initial
        self.P0 = P_prior_initial

        self.model_data_df = model_parameters_df.copy()
        self.model_data_df.insert(0, 'y', y_series)

        estimation_columns = ['a_prior', 'a_posterior', 'P_prior', 'P_posterior', 'v', 'F',
                              'F_inverse', 'K', 'L', 'a_hat', 'r', 'N', 'V', 'epsilon_hat',
                              'epsilon_hat_sigma2', 'eta_hat', 'eta_hat_sigma2', 'u', 'D']
        self.model_data_df = self.model_data_df.reindex(columns = self.model_data_df.columns.tolist()
                                                        + estimation_columns)
        self.model_data_df[estimation_columns] = pd.NA

        initial_index = self.model_data_df.index[0]
        self.n = len(y_series)
        self.m = self.Z[initial_index].shape[1]
        self.p = self.Z[initial_index].shape[0]

        self.a_prior[initial_index] = self.a0
        self.P_prior[initial_index] = self.P0

        self.filter_run = False
        self.smoother_run = False
        self.disturbance_smoother_run = False

    def filter(self):
        '''
        '''
        for index, key in enumerate(self.model_data_df.index):
            PZ = dot(self.P_prior[key], tr(self.Z[key]))
            self.F[key] = dot(self.Z[key], PZ) + self.H[key]

            v_shape = (self.p, 1)
            if self.is_all_missing(self.y[key]):
                self.Z[key] = zeros(self.Z[key].shape)
                self.v[key] = zeros(v_shape)
            else:
                if self.is_partial_missing(self.y[key]):
                    W = self.remove_missing_rows(identity(self.p), self.y[key])
                    self.y[key] = self.remove_missing_rows(self.y[key], self.y[key])
                    self.Z[key] = dot(W, self.Z[key])
                    self.d[key] = dot(W, self.d[key])
                    self.H[key] = dot(dot(W, self.H[key]), tr(W))
                self.v[key] = self.y[key] - dot(self.Z[key], self.a_prior[key]) - self.d[key]

            PZ = dot(self.P_prior[key], tr(self.Z[key]))
            F = dot(self.Z[key], PZ) + self.H[key]
            self.F_inverse[key] = inv(F)
            PZF_inv = dot(PZ, self.F_inverse[key])
            self.a_posterior[key] = self.a_prior[key] + dot(PZF_inv, self.v[key])
            self.P_posterior[key] = self.P_prior[key] - dot(PZF_inv, tr(PZ))

            a_prior = dot(self.T[key], self.a_posterior[key]) + self.c[key]
            P_prior = (dot(dot(self.T[key], self.P_posterior[key]), tr(self.T[key])) +
                       dot(dot(self.R[key], self.Q[key]), tr(self.R[key])))
            nxt_idx = index + 1
            try:
                nxt_key = self.model_data_df.index[nxt_idx]
            except IndexError:
                self.a_prior_final = a_prior
                self.P_prior_final = P_prior
            else:
                self.a_prior[nxt_key] = a_prior
                self.P_prior[nxt_key] = P_prior

        self.filter_run = True

    def smoother(self):
        '''
        '''
        if not self.filter_run:
            self.filter()

        self.r_final = zeros((self.m, 1))
        self.N_final = zeros((self.m, self.m))

        for index, key in reversed(list(enumerate(self.model_data_df.index))):
            ZF_inv = dot(tr(self.Z[key]), self.F_inverse[key])
            self.K[key] = dot(dot(self.T[key], self.P_prior[key]), ZF_inv)
            self.L[key] = self.T[key] - dot(self.K[key], self.Z[key])

            next_index = index + 1
            try:
                next_key = self.model_data_df.index[next_index]
            except IndexError:
                next_r = self.r_final
                next_N = self.N_final
            else:
                next_r = self.r[next_key]
                next_N = self.N[next_key]

            self.r[key] = dot(ZF_inv, self.v[key]) + dot(tr(self.L[key]), next_r)
            self.N[key] = (dot(ZF_inv, self.Z[key]) +
                           dot(dot(tr(self.L[key]), next_N), self.L[key]))

            self.a_hat[key] = self.a_prior[key] + dot(self.P_prior[key], self.r[key])
            self.V[key] = (self.P_prior[key] -
                           dot(dot(self.P_prior[key], self.N[key]), self.P_prior[key]))

        self.smoother_run = True

    def disturbance_smoother(self):
        '''
        '''
        if not self.smoother_run:
            self.smoother()

        for index, key in reversed(list(enumerate(self.model_data_df.index))):
            self.u[key] = dot(self.F_inverse[key], self.v[key])
            self.D[key] = self.F_inverse[key].copy()
            self.eta_hat_sigma2[key] = self.Q[key].copy()
            QR = dot(self.Q[key], tr(self.R[key]))

            next_index = index + 1
            try: 
                next_key = self.model_data_df.index[next_index]
            except IndexError:
                self.eta_hat[key] = dot(QR, self.r_final)
            else:
                self.u[key] = self.u[key] - dot(tr(self.K[key]), self.r[next_key])
                self.D[key] = self.D[key] + dot(dot(tr(self.K[key]), self.N[next_key]), self.K[key])

                self.eta_hat[key] = dot(QR, self.r[next_key])
                self.eta_hat_sigma2[key] = self.eta_hat_sigma2[key] - dot(dot(QR, self.N[next_key]), tr(QR))

            self.epsilon_hat[key] = dot(self.H[key], self.u[key])
            HDH = dot(dot(self.H[key], self.D[key]), self.H[key])
            self.epsilon_hat_sigma2[key] = self.H[key] - HDH

        self.disturbance_smoother_run = True

    def simulate_smoother(self):
        '''
        '''
        if not self.disturbance_smoother_run:
            self.disturbance_smoother()

        model_fields = ['Z', 'd', 'H', 'T', 'c', 'R', 'Q']
        model_data_df = self.model_data_df[model_fields]

        a0 = self.a0
        P0 = self.P0
        sim_df = self.simulate_model(model_data_df, a0, P0)
        for key in self.model_data_df.index:
            sim_df.loc[key,'y'] = self.copy_missing(sim_df.loc[key,'y'], self.y[key])

        sim_ssm = StateSpaceModel(sim_df['y'], model_data_df, a0, P0)
        sim_ssm.disturbance_smoother()

        sim_series_names = ['alpha', 'epsilon', 'eta']
        ssm_series_names = ['a_hat', 'epsilon_hat', 'eta_hat']
        estimation_error = pd.DataFrame()
        result_series = pd.DataFrame()

        for index, sim_col in enumerate(sim_series_names):
            ssm_col = ssm_series_names[index]
            estimation_error[sim_col] = sim_df[sim_col] - sim_ssm.model_data_df[ssm_col]
            result_series[sim_col] = estimation_error[sim_col] + self.model_data_df[ssm_col]

        return result_series

    def log_likelihood(self):
        '''
        '''
        if not self.filter_run:
            self.filter()

        term1 = self.n * self.p * log(2. * PI)
        term2 = self.model_data_df.apply(lambda df: log(det(df['F'])), axis=1).sum()
        term3 = self.model_data_df.apply(lambda df: dot(dot(tr(df['v']), df['F_inverse']), df['v']),
                                         axis=1).sum()[0, 0]
        result = -0.5 * (term1 + term2 + term3)

        return result

    @property
    def y(self):
        '''
        '''
        return self.model_data_df['y']

    @property
    def Z(self):
        '''
        '''
        return self.model_data_df['Z']

    @property
    def d(self):
        '''
        '''
        return self.model_data_df['d']

    @property
    def H(self):
        '''
        '''
        return self.model_data_df['H']

    @property
    def T(self):
        '''
        '''
        return self.model_data_df['T']

    @property
    def c(self):
        '''
        '''
        return self.model_data_df['c']

    @property
    def R(self):
        '''
        '''
        return self.model_data_df['R']

    @property
    def Q(self):
        '''
        '''
        return self.model_data_df['Q']

    @property
    def a_prior(self):
        '''
        '''
        return self.model_data_df['a_prior']

    @property
    def a_posterior(self):
        '''
        '''
        return self.model_data_df['a_posterior']

    @property
    def P_prior(self):
        '''
        '''
        return self.model_data_df['P_prior']

    @property
    def P_posterior(self):                               
        '''
        '''
        return self.model_data_df['P_posterior']

    @property
    def v(self):
        '''
        '''
        return self.model_data_df['v']

    @property
    def F(self):
        '''
        '''
        return self.model_data_df['F']

    @property
    def F_inverse(self):
        '''
        '''
        return self.model_data_df['F_inverse']

    @property
    def K(self):
        '''
        '''
        return self.model_data_df['K']

    @property
    def L(self):
        '''
        '''
        return self.model_data_df['L']

    @property
    def r(self):
        '''
        '''
        return self.model_data_df['r']

    @property
    def N(self):
        '''
        '''
        return self.model_data_df['N']

    @property
    def a_hat(self):
        '''
        '''
        return self.model_data_df['a_hat']

    @property
    def V(self):
        '''
        '''
        return self.model_data_df['V']

    @property
    def epsilon_hat(self):
        '''
        '''
        return self.model_data_df['epsilon_hat']

    @property
    def epsilon_hat_sigma2(self):
        '''
        '''
        return self.model_data_df['epsilon_hat_sigma2']

    @property
    def eta_hat(self):
        '''
        '''
        return self.model_data_df['eta_hat']

    @property
    def eta_hat_sigma2(self):
        '''
        '''
        return self.model_data_df['eta_hat_sigma2']

    @property
    def u(self):
        '''
        '''
        return self.model_data_df['u']

    @property
    def D(self):
        '''
        '''
        return self.model_data_df['D']

    @staticmethod
    def is_all_missing(value):
        '''
        '''
        try:
            if (value is pd.NA or
                value is None or
                isnan(value)):
                return True
        except (TypeError, ValueError):
             pass

        try:
            if all([StateSpaceModel.is_all_missing(val) for val in value]):
                return True
        except TypeError:
            pass

        return False

    @staticmethod
    def is_partial_missing(value):
        '''
        '''
        try:
            if any([StateSpaceModel.is_all_missing(val) for val in value]):
                return True
        except TypeError:
            pass

        return False

    @staticmethod
    def remove_missing_rows(value, ref):
        '''
        '''
        not_missing_mask = [not StateSpaceModel.is_all_missing(val) for val in ref]
        result = array(value)
        if len(result.shape) == 2:
            return result[not_missing_mask,:]
        return result[not_missing_mask]

    @staticmethod
    def copy_missing(value, ref):
        '''
        '''
        try:
            for index, val in enumerate(ref):
                if StateSpaceModel.is_all_missing(val):
                    try:
                        value[index][0] = pd.NA
                    except TypeError:
                        value[index] = pd.NA
        except TypeError:
            if StateSpaceModel.is_all_missing(ref):
                value = pd.NA

        return value
 
    @staticmethod
    def simulate_model(model_data_df, a0, P0):
        '''
        '''
        series_length = len(model_data_df)
        result_dict = {'y': [pd.NA] * series_length,
                       'alpha': [pd.NA] * series_length,
                       'epsilon': [pd.NA] * series_length,
                       'eta': [pd.NA] * series_length}
        result_df = pd.DataFrame(result_dict, index=model_data_df.index)

        for index, key in enumerate(model_data_df.index):
            prev_index = index - 1

            if prev_index == -1:
                curr_alpha = mv_norm(a0.ravel(), P0)
                curr_alpha.shape = (curr_alpha.shape[0], 1)
            else:
                prev_key = model_data_df.index[prev_index]
                prev_alpha          = result_df.loc[prev_key, 'alpha']
                curr_alpha_mu = (dot(model_data_df.loc[key, 'T'], prev_alpha) +
                                 model_data_df.loc[key, 'c'])
                curr_R = model_data_df.loc[key, 'R']
                curr_Q = model_data_df.loc[key, 'Q']
                curr_eta = mv_norm(zeros(curr_Q.shape[0]), curr_Q)
                curr_eta.shape = (curr_eta.shape[0], 1)
                curr_alpha = curr_alpha_mu + dot(curr_R, curr_eta)

                result_df.loc[prev_key, 'eta'] = curr_eta

            curr_H = model_data_df.loc[key, 'H']
            curr_Y_mu = (dot(model_data_df.loc[key, 'Z'], curr_alpha) +
                         model_data_df.loc[key, 'd'])
            curr_epsilon = mv_norm(zeros(curr_H.shape[0]), curr_H)
            curr_epsilon.shape = (curr_epsilon.shape[0], 1)
            curr_Y = curr_Y_mu + curr_epsilon

            result_df.loc[key, 'epsilon'] = curr_epsilon
            result_df.loc[key, 'alpha'] = curr_alpha
            result_df.loc[key, 'y'] = curr_Y

        return result_df
