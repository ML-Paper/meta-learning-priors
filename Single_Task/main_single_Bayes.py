
from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
from Utils import data_gen
from Utils.common import set_random_seed
from Single_Task import learn_single_Bayes

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels' ",
                    default='None')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # ConvNet3 / 'FcNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=300) # 300

parser.add_argument('--lr', type=float, help='learning rate (initial)',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

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

# Weights initialization:
prm.bayes_inits = {'Bayes-Mu': {'bias': 0, 'std': 0.1}, 'Bayes-log-var': {'bias': -10, 'std': 0.1}}

# None = use default initializer
# Note:
# 1. start with small sigma - so gradients variance estimate will be low

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

# prm.use_randomness_schedeule = True # False / True
# prm.randomness_init_epoch = 0
# prm.randomness_full_epoch = 500000000


#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
prm.lr_schedule = {} # No decay


# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'

# Generate task data set:
task_generator = data_gen.Task_Generator(prm)
limit_train_samples = None  # None
data_loader = task_generator.get_data_loader(prm, limit_train_samples=limit_train_samples)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

learn_single_Bayes.run_learning(data_loader, prm)