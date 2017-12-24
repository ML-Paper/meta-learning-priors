from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from Single_Task import learn_single_Bayes, learn_single_standard
from Utils.data_gen import Task_Generator
from Utils.common import write_result, set_random_seed


torch.backends.cudnn.benchmark = True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation: 'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='Permute_Labels')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=200)  # 200

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--log-file', type=str, help='Name of file to save log (default: no save)',
                    default='log')

prm = parser.parse_args()

prm.data_path = '../data'

set_random_seed(prm.seed)

n_experiments = 100  # 10

#  Define model:
prm.model_name = 'ConvNet3'

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10]}
prm.lr_schedule = {} # No decay


# For L2 regularization experiment:
prm_reg = deepcopy(prm)
prm.optim_args['weight_decay'] = 1e-3

# For freeze lower layers experiment:
prm_freeze = deepcopy(prm)
prm_freeze.not_freeze_list = ['fc_out']

# For bayes experiment -
# Weights initialization:
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}
prm.n_MC = 1 # Number of Monte-Carlo iterations
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'



task_generator = Task_Generator(prm)

test_err_orig = np.zeros(n_experiments)
test_err_scratch = np.zeros(n_experiments)
test_err_scratch_bayes = np.zeros(n_experiments)
test_err_transfer = np.zeros(n_experiments)
test_err_scratch_reg = np.zeros(n_experiments)
test_err_freeze = np.zeros(n_experiments)

for i in range(n_experiments):
    write_result('-' * 5 + ' Expirement #{} out of {}'.format(i+1, n_experiments), prm.log_file)

    # Generate the task #1 data set:
    task1_data = task_generator.get_data_loader(prm)
    n_samples_orig = task1_data['n_train_samples']

    #  Run learning of task 1
    write_result('-'*5 + 'Standard learning of task #1' + '-'*5, prm.log_file)
    test_err_orig[i], transfered_model = learn_single_standard.run_learning(task1_data, prm)

    # Generate the task 2 data set:
    limit_train_samples = 2000
    write_result('-'*5 + 'Generating task #2 with at most {} samples'.format(limit_train_samples) + '-'*5, prm.log_file)
    task2_data = task_generator.get_data_loader(prm, limit_train_samples = limit_train_samples)

    #  Run learning of task 2 from scratch:
    write_result('-'*5 + 'Standard learning of task #2 from scratch' + '-'*5, prm.log_file)
    test_err_scratch[i], _ = learn_single_standard.run_learning(task2_data, prm, verbose=0)


    #  Run Bayesian-learning of task 2 from scratch:
    write_result('-' * 5 + 'Bayesian learning of task #2 from scratch' + '-' * 5, prm.log_file)
    test_err_scratch_bayes[i], _ = learn_single_Bayes.run_learning(task2_data, prm, verbose=0)


    #  Run learning of task 2 using transferred initial point:
    write_result('-'*5 + 'Standard learning of task #2 using transferred weights as initial point' + '-'*5, prm.log_file)
    test_err_transfer[i], _ = learn_single_standard.run_learning(task2_data, prm, initial_model=transfered_model, verbose=0)

    #  Run learning of task 2 using transferred initial point:
    write_result('-'*5 + 'Standard learning of task #2 using transferred weights as initial point + freeze lower layers' + '-'*5, prm_freeze.log_file)
    test_err_freeze[i], _ = learn_single_standard.run_learning(task2_data, prm_freeze, initial_model=transfered_model, verbose=0)

    #  Run learning of task 2 from scratch + weight regularization:
    write_result('-' * 5 + 'Standard learning of task #2 from scratch' + '-' * 5, prm_reg.log_file)
    test_err_scratch_reg[i], _ = learn_single_standard.run_learning(task2_data, prm_reg, verbose=0)


write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
write_result('Averaging of {} experiments...'.format(n_experiments), prm.log_file)


write_result('Standard learning of task #1 ({} samples), average test error: {:.3}%, STD: {:.3}%'.
             format(n_samples_orig, 100*test_err_orig.mean(), 100*test_err_orig.std()), prm.log_file)

write_result('Standard learning of task #2  (at most {} samples)'
             ' from scratch, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch.mean(), 100*test_err_scratch.std()), prm.log_file)

write_result('Bayesian learning of task #2  (at most {} samples)'
             ' from scratch, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch_bayes.mean(), 100*test_err_scratch.std()), prm.log_file)


write_result('Standard learning of task #2  (at most {} samples) '
             'from scratch with L2 regularizer, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch_reg.mean(), 100*test_err_scratch_reg.std() ), prm_reg.log_file)

write_result('Standard learning of task #2  (at most {} samples)'
             ' using transferred weights as initial point, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_transfer.mean(), 100*test_err_transfer.std()), prm.log_file)

write_result('Standard learning of task #2  (at most {} samples) using transferred weights as initial point '
             ' + freeze lower layers, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_freeze.mean(), 100*test_err_freeze.std()), prm_freeze.log_file)
