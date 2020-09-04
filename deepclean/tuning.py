#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from hyperopt import hp, Trials


SCALE_PARAMS = ('beta_1', 'beta_2', 'decay', 'lr', 'reg')
LOC_PARAMS = ('dropout', 'epsilon', 'momentum', 'rho', 'recurrent_dropout')
DISCRETE_PARAMS = ('activation', 'optimizer',
                   'bias_initializer', 'kernel_initializer')

def gp_fit_1d(curr_params, curr_loss, search_space):
    """
    gaussian process regression

    Parameters:
    -----------
    curr_params: numpy.ndarray
        array of hyperparamters. shape: (n_trials, 1)
    curr_loss: numpy.ndarray
        array of losses. shape: (n_trials, )
    search_space: numpy.ndarray
        space to fit gaussian process. shape: (n_spaces, 1)

    Returns:
    --------
    mean: numpy.ndarry
        predicted mean loss at each point in search_space
    stdv: numpy.ndarry
        predicted stdv at each point in search_space
    """
    # TODO: add kernel to arguments
    kernel = C(1.0, (1e-8, 1e8)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(curr_params, curr_loss)

    # Get mean loss and stdv
    mean, stdv = gp.predict(search_space, return_std=True)

    return mean, stdv


def gp_suggest_1d(trials, key, search_space, n_sigma=3.0):
    """
    suggest next search parameters using a 1d gaussian process

    Parameters:
    -----------
    trials: dict
        dict with hyperparameters and losses
    key: str
        name of hyperparameters to fit
    search_space: numpy.ndarray
        space to fit gaussian process. shape: (n_spaces, 1)
    n_sigma: float

    Returns:
    --------
    next_params: numpy.ndarray
        suggested next parameters
    """
    curr_params = np.asarray(trials[key]).reshape(-1, 1)
    space = search_space[key]

    # only perform gp fit on non-discrete parameters
    if (key in SCALE_PARAMS or key in LOC_PARAMS) and isinstance(space, (np.ndarray, list, tuple)):

        curr_loss = trials['loss']

        n_trials = curr_params.shape[0]
        n_space = space.shape[0]

        # perform gaussian process
        if n_trials >= 2:

            # fit and predict mean and stdv
            mean, stdv = gp_fit_1d(curr_params, curr_loss, space)

            # calculate expected improvement
            min_loss = min(curr_loss)
            ei = min_loss - (mean - n_sigma*stdv)
            ei[ei < 0] = 0

            next_params = space[np.argmax(ei)]

        # randomly choose from space if not enough data
        else:
            next_params = space[np.random.randint(n_space)]

        # expect a len 1 array, convert to float
        next_params = float(next_params)

    # randomly choose discrete parameters
    elif key in DISCRETE_PARAMS and isinstance(space, (list, tuple)):
        next_params = np.random.choice(space)

    # copy the rest
    else:
        next_params = space

    return next_params


def create_space(params, mode, n=1000):
    """
    create  sample space
    scale params are sampled log-uniformly (Jeffrey's prior)
    loc params are sampled uniformly

    Parameters:
    -----------
    params: Dictionary
        dict with hyperparameters range
    mode: str
        tuning mode
    n: int
        resolution of sample space. only for gaussian process mode

    Returns
    space: Dictionary
        dict with hyperparameters space. Non-tunable or fixed hyperparameters are
        copied.
    """

    space = {}

    # begin create space
    for key, val in params.items():

        # create hyperopt space
        if mode == "random" or mode == "tpe":

            # log-uniform sampling scale parameters
            if key in SCALE_PARAMS and isinstance(val, (list, tuple)):
                log_v_min, log_v_max = np.log(min(val)), np.log(max(val))
                space[key] = hp.loguniform(key, log_v_min, log_v_max)

            # uniformly sampling loc parameters
            elif key in LOC_PARAMS and isinstance(val, (list, tuple)):
                v_min, v_max = min(val), max(val)
                space[key] = hp.uniform(key, v_min, v_max)

            # randomly sampling discrete params
            elif key in DISCRETE_PARAMS and isinstance(val, (list, tuple)):
                space[key] = hp.choice(key, val)

            # copy the rest
            else:
                space[key] = val

        # create linear space for gaussian process
        elif mode == "gp":

            # linear space for both scale and loc parameters
            if (key in SCALE_PARAMS or key in LOC_PARAMS) and isinstance(val, (list, tuple)):
                space[key] = np.linspace(min(val), max(val), n).reshape(-1, 1)

            # copy discrete and fixed paramters
            else:
                space[key] = val

    return space


def save_tuning_results(
    savedir     = 'Tuning',
    loop        = 'Loop_0',
    trials      = None
    ):
    """
    save best result

    Parameters
    ----------
    savedir : str
        directory to save
    loop: str
        current network loop
    trials: hyperopt.Trials or dict
        hyperopt trials object
    """

    # save best hyperparameters and best loss
    if isinstance(trials, Trials):
        best_loss = trials.best_trial['result']['loss']
        best_params = trials.best_trial['misc']['vals']
    else:
        indx = np.argmin(trials['loss'])
        best_params = {}
        for k, v in trials.items():
            if k in SCALE_PARAMS or k in LOC_PARAMS or k in DISCRETE_PARAMS:
                best_params[k] = v[indx]
        best_loss = trials['loss'][indx]

    with open('{0}/{1}_res.txt'.format(savedir, loop), 'w') as f:
        f.write('LOSS {}\n'.format(best_loss))
        for k, v in best_params.items():
            f.write('{0} = {1}\n'.format(k, v))
        f.write('\n')

    # save trials object
    with open('{0}/{1}_trials.pkl'.format(savedir, loop), 'wb') as f:
        pickle.dump(trials, f, protocol=-1)
