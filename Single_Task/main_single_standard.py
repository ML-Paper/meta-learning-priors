
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.optim as optim

from Utils import data_gen
from Utils.common import set_random_seed, save_model_state
from Single_Task import learn_single_standard

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'",
                    default='None')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # ConvNet3 / 'FcNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=128)

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
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #  'weight_decay':5e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9, 'weight_decay':5e-4}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [150, 225]}
prm.lr_schedule = {} # No decay

# Generate task data set:
limit_train_samples = None  # None
task_generator = data_gen.Task_Generator(prm)
data_loader = task_generator.get_data_loader(prm, limit_train_samples=limit_train_samples)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

test_err, model = learn_single_standard.run_learning(data_loader, prm)

dir_path = './saved'
f_name='standard_weights'
f_path = save_model_state(model, dir_path, name=f_name)
print('Trained model saved in ' + f_path)