'''
Created on 19 Apr 2020

@author: adriandickeson
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sstspack import StateSpaceModel as SSM, modeldata as md

def get_model_and_sim(y, Q, H):
    length = len(y)
    data_df = md.get_local_level_model_data(length, Q, H)
    ssm = SSM(y, data_df, np.zeros((1,1)), np.ones((1,1)))
    ssm.filter()
    ssm.smoother()
    ssm.disturbance_smoother()
    sim = ssm.simulate_smoother()
    return ssm, sim

def plot_sim(data, ssm, sim, title):
    non_missing_mask1 = [not SSM.is_all_missing(value) for value in data['Observed 1']]
    non_missing_mask2 = [not SSM.is_all_missing(value) for value in data['Observed 2']]

    fig, ax = plt.subplots(1)
    ax.scatter(data.index[non_missing_mask1], data['Observed 1'][non_missing_mask1], marker='x')
    ax.scatter(data.index[non_missing_mask2], data['Observed 2'][non_missing_mask2], marker='x')
    ax.plot(ssm.a_hat)
    ax.plot(sim['alpha'])
    fig.suptitle(title)

if __name__ == '__main__':
    fn_path = 'data/noisy_cos_data.csv'
    data = pd.read_csv(fn_path)
    y = data.apply(lambda x: np.array([[x['Observed 1']], [x['Observed 2']]]), axis=1)
    alpha = data['Actual']

    epsilon = y - alpha
    eta = np.array(alpha)[1:] - np.array(alpha)[:-1]
    H = np.array([[0.1 ** 2, 0.], [0., 0.2 ** 2]])
    Q = np.std(eta) ** 2

    ssm, sim = get_model_and_sim(y, Q, H)
    plot_sim(data, ssm, sim, 'Complete Data')

    data['Observed 1'][21:40] = pd.NA
    data['Observed 2'][31:50] = pd.NA
    y = data.apply(lambda x: np.array([[x['Observed 1']], [x['Observed 2']]]), axis=1)

    ssm, sim = get_model_and_sim(y, Q, H)
    plot_sim(data, ssm, sim, 'Missing Data')

    plt.show()
