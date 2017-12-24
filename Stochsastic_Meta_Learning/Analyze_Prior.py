
from __future__ import absolute_import, division, print_function

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from Models.stochastic_models import get_model
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

# -------------------------------------------------------------------------------------------
# Auxilary functions:
#----------------------------------------------------------------------------------------------

def extract_param_list(model, name1, name2):
    return [named_param for named_param in model.named_parameters() if name1 in named_param[0] and name2 in named_param[0]]

def log_var_to_sigma(log_var_params):
    return [(named_param[0].replace('_log_var', '_sigma'),
              0.5 * torch.exp(named_param[1]))
             for named_param in log_var_params]

def get_params_statistics(param_list):
    n_list = len(param_list)
    mean_list = np.zeros(n_list)
    std_list = np.zeros(n_list)
    for i_param, named_param in enumerate(param_list):
        param_name = named_param[0]
        param_vals = named_param[1]
        param_mean = param_vals.mean().data[0]
        param_std = param_vals.std().data[0]
        mean_list[i_param] = param_mean
        std_list[i_param] = param_std
        print('Parameter name: {}, mean value: {:.3}, STD: {:.3}'.format(param_name, param_mean, param_std))
    return mean_list, std_list


def plot_statistics(mean_list, std_list, name):
    plt.figure()
    n_list = len(mean_list)
    plt.errorbar(range(n_list), mean_list, yerr=std_list)
    # plt.title("Statistics of the prior {} ".format(name))
    plt.xticks(np.arange(n_list))
    plt.xlabel('Layer')
    plt.ylabel(name)

# -------------------------------------------------------------------------------------------
# Analysis function:
#----------------------------------------------------------------------------------------------

def run_prior_analysis(prior_model, layers_names=None, showPlt=True):

    # w_mu_params = extract_param_list(prior_model,'_mean', '.w_')
    # b_mu_params = extract_param_list(prior_model,'_mean', '.b_')
    w_log_var_params = extract_param_list(prior_model,'_log_var', '.w_')
    # b_log_var_params = extract_param_list(prior_model,'_log_var', '.b_')

    n_layers = len(w_log_var_params)

    # w_sigma_params = log_var_to_sigma(w_log_var_params)
    # b_sigma_params = log_var_to_sigma(b_log_var_params)

    plot_statistics(*get_params_statistics(w_log_var_params), name=r'$\log (\sigma^2)$')
    layers_inds = np.arange(n_layers)
    if layers_names:
        layers_names = [str(i) + ' (' + layers_names[i] + ')' for i in layers_inds]
    else:
        layers_names = [str(i) for i in layers_inds]

    plt.xticks(layers_inds, layers_names)
    if showPlt:
        plt.show()



# -------------------------------------------------------------------------------------------
# execute only if run as a script
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":


    # font = {'family' : 'normal',
    #         'weight' : 'normal',
    #         'size'   : 15}
    # matplotlib.rc('font', **font)

    # settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                        default='MNIST')

    parser.add_argument('--seed', type=int,  help='random seed',
                        default=1)

    parser.add_argument('--log-file', type=str, help='Name of file to save log (default: no save)',
                        default='log')

    prm = parser.parse_args()

    set_random_seed(prm.seed)


    # Weights initialization (for Bayesian net):
    prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}

    #
    #  Load pre-trained prior
    #

    dir_path = './saved'

    name = 'Permuted_pixels'  # 'Permuted_pixels' / Permuted_labels

    if name == 'Permuted_pixels':
        # Permute Pixels:
        file_name_prior = 'prior_New_PermutePixels'
        prm.model_name = 'FcNet3'
        layers_names = ('FC1', 'FC2', 'FC3', 'FC_out')
        # ***************
    else:
        # Permute Labels:
        file_name_prior = 'pror_Permuted_labels_MNIST'
        prm.model_name = 'ConvNet'
        layers_names = ('conv1', 'conv2', 'FC1', 'FC_out')

    full_path = os.path.join(dir_path, file_name_prior)

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    # Then load the weights:
    is_loaded = load_model_state(prior_model, dir_path, name=file_name_prior)
    if not is_loaded:
        raise ValueError('No prior found in the path: ' + full_path)
    print('Pre-trained  prior loaded from ' + full_path)

    run_prior_analysis(prior_model, layers_names, showPlt=True)
