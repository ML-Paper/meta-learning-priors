
from __future__ import absolute_import, division, print_function

import argparse
import timeit, time
import numpy as np
import torch
import torch.optim as optim

from MAML import meta_train_MAML_finite_tasks, meta_test_MAML, meta_train_MAML_infinite_tasks
from Models.deterministic_models import get_model
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, write_result, set_random_seed

torch.backends.cudnn.benchmark = True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help='Data set',
                    default='Omniglot') # 'MNIST' / 'Omniglot'

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=0)

parser.add_argument('--data-transform', type=str, help="Data transformation",
                    default='Rotate90') #  'None' / 'Permute_Pixels' / 'Permute_Labels' / Rotate90

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='OmConvNet')  # ConvNet3 / 'FcNet3' / 'OmConvNet'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=5)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

# MAML hyper-parameters:
parser.add_argument('--alpha', type=float, help='Step size for gradient step',
                    default=0.4)
parser.add_argument('--n_meta_train_grad_steps', type=int, help='Number of gradient steps in meta-training',
                    default=1)
parser.add_argument('--n_meta_train_iterations', type=int, help='number of iterations in meta-training',
                    default=60000)  # 15000
parser.add_argument('--n_meta_test_grad_steps', type=int, help='Number of gradient steps in meta-testing',
                    default=3)
parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=32)
# Run parameters:
parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'
parser.add_argument('--meta_model_file_name', type=str, help='File name to save meta-model or to load from',
                    default='meta_model')

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training sampels in the meta-test tasks (0 = unlimited)',
                    default=0)
parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation',
                    default=100)

parser.add_argument('--log-file', type=str, help='Name of file to save log (None = no save)',
                    default='log')


# Omniglot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)
parser.add_argument('--K_Shot', type=int, help='Number of training sample per class (for Omniglot)',
                    default=5)  # Note: number of test samples per class is 20-K (the rest of the data)
parser.add_argument('--chars_split_type', type=str, help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')
parser.add_argument('--n_meta_train_chars', type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)


prm = parser.parse_args()
prm.data_path = '../data'

set_random_seed(prm.seed)

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {} # No decay


dir_path = './saved'

task_generator = Task_Generator(prm)
# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

if prm.mode == 'MetaTrain':

    n_train_tasks = prm.n_train_tasks
    if n_train_tasks:
        # In this case we generate a finite set of train (observed) task before meta-training.

        # Generate the data sets of the training-tasks:
        write_result('-' * 5 + 'Generating {} training-tasks'.format(n_train_tasks) + '-' * 5, prm.log_file)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, meta_split='meta_train')

        # Meta-training to learn meta-model (theta params):
        meta_model = meta_train_MAML_finite_tasks.run_meta_learning(train_data_loaders, prm)
    else:
        # In this case we observe new tasks generated from the task-distribution in each meta-iteration.
        write_result('-' * 5 + 'Infinite train tasks - New training tasks are drawn from tasks distribution in each iteration...' + '-' * 5, prm.log_file)

        # Meta-training to learn meta-model (theta params):
        meta_model = meta_train_MAML_infinite_tasks.run_meta_learning(prm, task_generator)

    # save learned meta-model:
    f_path = save_model_state(meta_model, dir_path, name=prm.meta_model_file_name)
    print('Trained meta-model saved in ' + f_path)


elif prm.mode == 'LoadMetaModel':

    # Loads  previously training prior.
    # First, create the model:
    meta_model = get_model(prm)
    # Then load the weights:
    load_model_state(meta_model, dir_path, name=prm.meta_model_file_name)
    print('Pre-trained  meta-model loaded from ' + dir_path)
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
write_result('Meta-Testing with transferred meta-params....', prm.log_file)

test_err_bayes = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_bayes[i_task], _ = meta_test_MAML.run_learning(task_data, meta_model, prm, verbose=0)


# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------

# write_result('Run standard learning from scratch....', prm.log_file)
#
# test_err_standard = np.zeros(n_test_tasks)
# for i_task in range(n_test_tasks):
#     print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
#     task_data = test_tasks_data[i_task]
#     test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm, verbose=0)
#

# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
write_result('-'*5 + ' Final Results: '+'-'*5, prm.log_file)
write_result('Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_bayes.mean(), 100 * test_err_bayes.std()), prm.log_file)
# write_result('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
#              format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm.log_file)
#
stop_time = timeit.default_timer()
write_result('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm.log_file)