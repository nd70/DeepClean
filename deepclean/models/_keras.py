import re
import sys

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras import optimizers
from keras.layers import (Dense, Dropout, LSTM, Flatten, BatchNormalization)
from keras.callbacks import TensorBoard

def create_callbacks(logDir, batch_size):
    """ Create Tensorboard callback """
    tb_callback = TensorBoard(
        log_dir = 'Tensorboard/{}/keras'.format(logDir),
        histogram_freq = 1,
        batch_size  = batch_size,
        write_graph = True,
        write_grads = True,
        )
    return tb_callback

def get_model(input_shape       = None,
              recurrent_activation     = 'tanh',
              recurrent_initializer    = 'glorot_uniform',
              dense_activation         = 'relu',
              dense_kernel_initializer = 'glorot_uniform',
              dense_bias_initializer   = 'ones',
              dropout           = 0.1,
              recurrent_dropout = 0.0):

    model = Sequential()
    model.add(LSTM(16,
                   input_shape        = input_shape,
                   return_sequences   = True,
                   activation         = recurrent_activation,
                   dropout            = dropout,
                   kernel_initializer = recurrent_initializer,
                   bias_initializer   = recurrent_initializer,
                   recurrent_dropout  = recurrent_dropout))
    model.add(BatchNormalization(axis=-1))
    model.add(LSTM(6,
                   return_sequences   = False,
                   activation         = recurrent_activation,
                   dropout            = dropout,
                   kernel_initializer = recurrent_initializer,
                   bias_initializer   = recurrent_initializer,
                   recurrent_dropout  = recurrent_dropout))
    model.add(BatchNormalization(axis=-1))

    for _ in range(2):
        model.add(Dense(8,
                        activation         = dense_activation,
                        kernel_initializer = dense_kernel_initializer,
                        bias_initializer   = dense_bias_initializer,))
    model.add(Dense(1,
                    kernel_initializer = dense_kernel_initializer,
                    bias_initializer   = dense_bias_initializer,))
    return model


def get_optimizer(opt,
                  decay    = None,
                  lr       = None,
                  momentum = 0.0,
                  nesterov = False,
                  beta_1   = 0.9,
                  beta_2   = 0.999,
                  epsilon  = 1e-8,
                  rho      = None):

    """
    get_optimizer is a wrapper for Keras optimizers.

    Parameters
    ----------
    beta_1 : `float`
        adam optimizer parameter in range [0, 1) for updating bias first
        moment estimate
    beta_2 : `float`
        adam optimizer parameter in range [0, 1) for updating bias second
        moment estimate
    decay : `None` or `float`
        learning rate decay
    epsilon : `float`
        parameter for numerical stability
    opt : `str`
        Keras optimizer. Options: "sgd", "adam", "nadam", "rmsprop",
        "adagrad", "adamax" and "adadelta"
    lr : `None` or `float`
        optimizer learning rate
    momentum : `float`
        accelerate the gradient descent in the direction that dampens
        oscillations
    nesterov : `bool`
        use Nesterov Momentum
    rho : `None` or `float`
        gradient history

    Returns
    -------
    optimizer : :class:`keras.optimizer`
        keras optimizer object
    """

    ###############################
    # Stochastic Gradient Descent #
    ###############################
    if opt == 'sgd':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.SGD(lr       = lr,
                                   momentum = momentum,
                                   decay    = decay,
                                   nesterov = nesterov)
    ########
    # Adam #
    ########
    elif opt == 'adam':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    ##########
    # Adamax #
    ##########
    elif opt == 'adamax':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    #########
    # Nadam #
    #########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'nadam':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.004

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)

    ###########
    # RMSprop #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values (except the learning
    # rate, which can be freely tuned).
    elif opt == 'rmsprop':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.9

        optimizer = optimizers.RMSprop(lr      = lr,
                                       rho     = rho,
                                       epsilon = epsilon,
                                       decay   = decay)
    ###########
    # Adagrad #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adagrad':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adagrad(lr      = lr,
                                       decay   = decay,
                                       epsilon = epsilon)

    ############
    # Adadelta #
    ############
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adadelta':

        if lr is None:
            lr = 1.0

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.95

        optimizer = optimizers.Adadelta(lr      = lr,
                                        rho     = rho,
                                        epsilon = epsilon,
                                        decay   = decay)
    else:
        print('ERROR: Unknown optimizer')
        sys.exit(1)

    return optimizer


# convienient function


def create_model(

    input_shape = (None, None),

    # network structure
    recurrent_activation     = 'tanh',
    recurrent_initializer    = 'glorot_uniform',
    dense_activation         = 'relu',
    dense_kernel_initializer = 'glorot_uniform',
    dense_bias_initializer   = 'zeros',
    dropout           = 0.1,
    recurrent_dropout = 0.0,

    # loss function
    loss        = 'psd',

    # training
    optimizer   = 'adam',
    lr          = None,
    momentum    = 0.0,
    nesterov    = False,
    beta_1      = 0.9,
    beta_2      = 0.999,
    epsilon     = 1e-8,
    rho         = None,
    decay       = None,
    **kwargs
    ):

    # build network architecture and compile
    model = get_model(input_shape       = input_shape,
                      recurrent_activation     = recurrent_activation,
                      recurrent_initializer    = recurrent_initializer,
                      dense_activation         = dense_activation,
                      dense_kernel_initializer = dense_kernel_initializer,
                      dense_bias_initializer   = dense_bias_initializer,
                      dropout           = dropout,
                      recurrent_dropout = recurrent_dropout,
                      )

    optimizer = get_optimizer(optimizer,
                              decay    = decay,
                              lr       = lr,
                              momentum = momentum,
                              nesterov = nesterov,
                              beta_1   = beta_1,
                              beta_2   = beta_2,
                              epsilon  = epsilon,
                              rho      = rho)

    model.compile(loss=loss, optimizer=optimizer)
    return model
