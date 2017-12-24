#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import data_gen

from Models.stochastic_layers import StochasticLinear, StochasticConv2d, StochasticLayer



# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------
def get_size_of_conv_output(input_shape, conv_func):
    # generate dummy input sample and forward to get shape after conv layers
    batch_size = 1
    input = Variable(torch.rand(batch_size, *input_shape))
    output_feat = conv_func(input)
    conv_out_size = output_feat.data.view(batch_size, -1).size(1)
    return conv_out_size

#
# -------------------------------------------------------------------------------------------
#  Main function
#  -------------------------------------------------------------------------------------------
def get_model(prm, model_type='Stochastic'):

    model_name = prm.model_name

    # Get task info:
    task_info = data_gen.get_info(prm)

    # Define default layers functions
    def linear_layer(in_dim, out_dim):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, prm)

    def conv2d_layer(in_channels, out_channels, kernel_size, use_bias=False, stride=1, padding=0, dilation=1):
        if model_type == 'Standard':
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        elif model_type == 'Stochastic':
            return StochasticConv2d(in_channels, out_channels, kernel_size, prm, use_bias, stride, padding, dilation)

    #  Return selected model:
    if model_name == 'FcNet3':
        model = FcNet3(model_type, model_name, linear_layer, conv2d_layer, task_info)

    elif model_name == 'ConvNet3':
        model = ConvNet3(model_type, model_name, linear_layer, conv2d_layer, task_info)

    elif model_name == 'BayesDenseNet':
        from Models.densenetBayes import get_bayes_densenet_model_class
        densenet_model = get_bayes_densenet_model_class(prm, task_info)
        model = densenet_model(depth=20)

    else:
        raise ValueError('Invalid model_name')

    model.cuda()

    # For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)
    if hasattr(prm, 'override_eps_std'):
        model.set_eps_std(prm.override_eps_std) # debug
    return model


#  -------------------------------------------------------------------------------------------
#   Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class general_model(nn.Module):
    def __init__(self):
        super(general_model, self).__init__()

    def set_eps_std(self, eps_std):
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

# -------------------------------------------------------------------------------------------
# Models collection
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(general_model):
    def __init__(self, model_type, model_name, linear_layer, conv2d_layer, task_info):
        super(FcNet3, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        input_shape = task_info['input_shape']
        n_classes = task_info['n_classes']
        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 400
        n_hidden3 = 400
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc3 = linear_layer(n_hidden2, n_hidden3)
        self.fc_out = linear_layer(n_hidden3, n_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # flatten image
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc_out(x)
        return x

# -------------------------------------------------------------------------------------------
#  ConvNet
# -------------------------------------------------------------------------------- -----------
class ConvNet3(general_model):
    def __init__(self, model_type, model_name, linear_layer, conv2d_layer, task_info):
        super(ConvNet3, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        input_shape = task_info['input_shape']
        color_channels = input_shape[0]
        n_classes = task_info['n_classes']
        n_filt1 = 10
        n_filt2 = 20
        n_hidden_fc1 = 50
        self.conv1 = conv2d_layer(color_channels, n_filt1, kernel_size=5)
        self.conv2 = conv2d_layer(n_filt1, n_filt2, kernel_size=5)
        conv_feat_size = get_size_of_conv_output(input_shape, self._forward_features)
        self.fc1 = linear_layer(conv_feat_size, n_hidden_fc1)
        self.fc_out = linear_layer(n_hidden_fc1, n_classes)

    def _forward_features(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), 2))
        x = F.elu(F.max_pool2d(self.conv2(x), 2))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_out(x)
        return x