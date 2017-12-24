
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import argparse
import timeit, time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from Stochsastic_Meta_Learning import meta_test_Bayes, meta_train_Bayes_finite_tasks, meta_train_Bayes_infinite_tasks

from Models import stochastic_models, deterministic_models
from Single_Task import learn_single_standard
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed, zeros_gpu
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from Models.stochastic_layers import StochasticLayer

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help='Data set',
                    default='MNIST') # 'MNIST' / 'Omniglot'

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=5)

parser.add_argument('--data-transform', type=str, help="Data transformation",
                    default='Permute_Labels') #  'None' / 'Permute_Pixels' / 'Permute_Labels' / Rotate90

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # ConvNet3 / 'FcNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num_epochs', type=int, help='number of epochs to train',
                    default=50)  # 10 / 100

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=200)  # 10 / 300

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=5)
# Run parameters:
parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--meta_model_file_name', type=str, help='File name to save meta-model or to load from',
                    default='meta_model')

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training sampels in the meta-test tasks (0 = unlimited)',
                    default=2000)

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation',
                    default=100)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')

# # Omniglot Parameters:
# parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
#                     default=5)
# parser.add_argument('--K_Shot', type=int, help='Number of training sample per class (for Omniglot)',
#                     default=5)  # Note: number of test samples per class is 20-K (the rest of the data)
# parser.add_argument('--chars_split_type', type=str, help='how to split the Omniglot characters  - "random" / "predefined_split"',
#                     default='random')
# parser.add_argument('--n_meta_train_chars', type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
#                     default=1200)

prm = parser.parse_args()

prm.data_path = '../data'

set_random_seed(prm.seed)


# Weights initialization (for Bayesian net):
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}
# Note:
# 1. start with small sigma - so gradients variance estimate will be low
# 2.  don't init with too much std so that complexity term won't be too large

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay


# MPB alg  params:
prm.complexity_type = 'NewBoundSeeger'
#  'Variational_Bayes' / 'PAC_Bayes_McAllaster' / 'PAC_Bayes_Pentina' / 'PAC_Bayes_Seeger'  / 'KLD' / 'NoComplexity' /  NewBound / NewBoundSeeger
prm.kappa_prior = 2e3  #  parameter of the hyper-prior regularization
prm.kappa_post = 1e-3  # The STD of the 'noise' added to prior
prm.delta = 0.1  #  maximal probability that the bound does not hold

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

dir_path = './saved'

task_generator = Task_Generator(prm)
# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

if prm.mode == 'MetaTrain':

    n_train_tasks = prm.n_train_tasks
    # In this case we generate a finite set of train (observed) task before meta-training.
    # Generate the data sets of the training tasks:
    write_result('-' * 5 + 'Generating {} training-tasks'.format(n_train_tasks) + '-' * 5, prm.log_file)
    train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, meta_split='meta_train')


    # Run standard learning for each task and average the parameters:
    for i_task in range(n_train_tasks):
        print('Learning train-task {} out of {}'.format(i_task+1, n_train_tasks))
        data_loader = train_data_loaders[i_task]
        test_err, curr_model = learn_single_standard.run_learning(data_loader, prm, verbose=0)
        if i_task == 0:
            avg_param_vec = parameters_to_vector(curr_model.parameters()) * (1 / n_train_tasks)
        else:
            avg_param_vec += parameters_to_vector(curr_model.parameters()) * (1 / n_train_tasks)

    avg_model = deterministic_models.get_model(prm)
    vector_to_parameters(avg_param_vec, avg_model.parameters())

    # create the prior model:
    prior_model = stochastic_models.get_model(prm)
    prior_layers_list = [layer for layer in prior_model.modules() if isinstance(layer, StochasticLayer)]
    avg_model_layers_list = [layer for layer in avg_model.modules()
                             if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)]
    assert len(avg_model_layers_list)==len(prior_layers_list), "lists not equal"

    for i_layer, prior_layer in enumerate(prior_layers_list):
        if hasattr(prior_layer, 'w'):
            prior_layer.w['log_var'] = torch.nn.Parameter(zeros_gpu(1))
            prior_layer.w['mean'] = avg_model_layers_list[i_layer].weight
        if hasattr(prior_layer, 'b'):
            prior_layer.b['log_var'] = torch.nn.Parameter(zeros_gpu(1))
            prior_layer.b['mean'] = avg_model_layers_list[i_layer].bias


    # save learned prior:
    f_path = save_model_state(prior_model, dir_path, name=prm.meta_model_file_name)
    print('Trained prior saved in ' + f_path)


elif prm.mode == 'LoadMetaModel':

    # Loads  previously training prior.
    # First, create the model:
    prior_model = stochastic_models.get_model(prm)
    # Then load the weights:
    load_model_state(prior_model, dir_path, name=prm.meta_model_file_name)
    print('Pre-trained  prior loaded from ' + dir_path)
else:
    raise ValueError('Invalid mode')

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = prm.n_test_tasks

limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks
if limit_train_samples_in_test_tasks == 0:
    limit_train_samples_in_test_tasks = None

write_result('-'*5 + 'Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples_in_test_tasks)+'-'*5, prm.log_file)


test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test',
                                                   limit_train_samples=limit_train_samples_in_test_tasks)
#
# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
write_result('Meta-Testing with transferred prior....', prm.log_file)

test_err_bayes = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_bayes[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model, prm, init_from_prior, verbose=0)


# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------

write_result('Run standard learning from scratch....', prm.log_file)
test_err_standard = np.zeros(n_test_tasks)
prm_standard = deepcopy(prm)
prm_standard.num_epochs = prm.n_meta_test_epochs
for i_task in range(n_test_tasks):
    print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm_standard, verbose=0)


# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
#  Print prior analysis
# -------------------------------------------------------------------------------------------
# from Stochsastic_Meta_Learning.Analyze_Prior import run_prior_analysis
# run_prior_analysis(prior_model)

stop_time = timeit.default_timer()
write_result('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm.log_file)
# -------------------------------------------------------------------------------------------
write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
write_result('Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_bayes.mean(), 100 * test_err_bayes.std()), prm.log_file)
write_result('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
             format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm.log_file)
