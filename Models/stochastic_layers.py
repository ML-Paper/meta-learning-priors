from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils.common import randn_gpu

# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLayer(nn.Module):
    # base class of stochastic layers with re-parametrization
    # self.init  and self.operation should be filled by derived classes

    def init_stochastic_layer(self, weights_size, bias_size, prm):

        inits = prm.bayes_inits
        mu_bias = inits['Bayes-Mu']['bias']
        mu_std = inits['Bayes-Mu']['std']
        log_var_bias = inits['Bayes-log-var']['bias']
        log_var_std = inits['Bayes-log-var']['std']

        self.w_mu = get_randn_param(weights_size, mu_bias, mu_std)
        self.w_log_var = get_randn_param(weights_size, log_var_bias, log_var_std)
        self.w = {'mean': self.w_mu, 'log_var': self.w_log_var}

        if bias_size is not None:
            self.b_mu = get_randn_param(bias_size, mu_bias, mu_std)
            self.b_log_var = get_randn_param(bias_size, log_var_bias, log_var_std)
            self.b = {'mean': self.b_mu, 'log_var': self.b_log_var}


    def forward(self, x):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)
        # self.operation should be linear or conv

        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b['mean']
        else:
            b_var = None
            bias_mean = None

        out_mean = self.operation(x, self.w['mean'], bias=bias_mean)

        eps_std = self.eps_std
        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            out_var = self.operation(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = randn_gpu(size=out_mean.size(), mean=0, std=eps_std)

            noise = Variable(noise, requires_grad=False)

            out_var = F.relu(out_var) # to avoid nan due to numerical errors
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    def set_eps_std(self, eps_std):
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLinear(StochasticLayer):


    def __init__(self, in_dim, out_dim, prm, use_bias=True):
        super(StochasticLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        if use_bias:
            bias_size = out_dim
        else:
            bias_size = None
        self.init_stochastic_layer(weights_size, bias_size, prm)
        self.eps_std = 1.0


    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)

# -------------------------------------------------------------------------------------------
#  Stochastic conv2d layer
# -------------------------------------------------------------------------------------------

class StochasticConv2d(StochasticLayer):

    def __init__(self, in_channels, out_channels, kernel_size, prm, use_bias=False, stride=1, padding=0, dilation=1):
        super(StochasticConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        kernel_size = make_pair(kernel_size)
        self.kernel_size = kernel_size

        weights_size = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        if use_bias:
            bias_size = (out_channels)
        else:
            bias_size = None
        self.init_stochastic_layer(weights_size, bias_size, prm)
        self.eps_std = 1.0


    def __str__(self):
        return 'StochasticConv2d({} -> {}, kernel_size={})'.format(self.in_channels, self.out_channels, self.kernel_size)

    def operation(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation)


# -------------------------------------------------------------------------------------------
#  Auxilary functions
# -------------------------------------------------------------------------------------------
def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x

# def get_randn_param(shape, mean, std):
#     return nn.Parameter(randn_gpu(shape, mean, std))


def get_randn_param(shape, mean, std):
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.FloatTensor(*shape).normal_(mean, std))
