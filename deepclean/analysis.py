#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
# Hush tensorflow AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

import deepclean.preprocessing as ppr
import scipy.io as sio
import scipy.signal as sig
import seaborn as sns
import sys
from cycler import cycler
from configparser import ConfigParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def set_plot_style():
    """
    provide matplotlib plotting style
    """
    plt.style.use('bmh')
    matplotlib.rcParams.update({
        'axes.grid': True,
        'axes.titlesize': 'medium',
        'font.family': 'serif',
        'font.size': 12,
        'grid.color': 'xkcd:grey',
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.linewidth': 1,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 13,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.1,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{txfonts}'
    })

    matplotlib.rc("savefig", dpi=100)
    matplotlib.rc("figure", figsize=(7, 4))


def plot_psd(target, prediction,
             fs      = 512,
             nfft    = 4096,
             plotDir = 'Plots',
             title   = 'Neural Network Validation PSD',
             saveas  = 'validation_psd',
             savepdf = False,
             fmin    = 4,
             fmax    = 256):
    """
    plot the psd of the DARM (testing) timeseries, the prediction timeseries
    calculated by the network, and the residual

    Parameters
    ----------
    target : `numpy.ndarray`
        validation DARM timeseries
    prediction: `numpy.ndarray`
        validation prediction timeseries
    fs : `int`
        data sample rate
    nfft : `int`
        overlapping windows
    plotDir : `str`
        path to store plots
    title : `str`
        plot title
    saveas : `str`
        output filename for plot
    savepdf : `bool`
        when set to True, both png and pdf filetypes will be saved
    fmin : `float`
        x-axis min
    fmax : `float`
        x-axis max
    """

    residual = target - prediction
    ff1, pp = sig.welch([target, prediction, residual],
                        fs      = fs,
                        nperseg = nfft,
                        axis    = -1)

    strain = np.sqrt(pp).T
    ff2, co = sig.coherence(target, prediction,
                            fs      = fs,
                            nperseg = nfft)

    # make plots to evaluate success / failure of the regression
    fig, (ax1, ax2) = plt.subplots(nrows       = 2,
                                   sharex      = True,
                                   figsize     = (8, 6),
                                   gridspec_kw = {'height_ratios': [2, 1]})

    ax1.set_prop_cycle(cycler('color', ['xkcd:black',
                                        'xkcd:pumpkin',
                                        'xkcd:twilight blue']))

    labels = ['DARM', 'Prediction', 'Subtracted']
    linestyles = ['-', '--', '-']
    for i in range(len(labels)):
        ax1.loglog(ff1.T, strain[:,i],
                   alpha      = 0.8,
                   rasterized = True,
                   linestyle  = linestyles[i],
                   label      = labels[i])

    ax1.legend(fontsize='small', loc=1)
    ax1.set_xlim([fmin, fmax])
    # ax1.set_ylim([1e-25, 5e-18])
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
    ax1.set_title(title)

    # Plot the ratio of the PSDs. Max ratio where DARM = subtracted
    ax2.loglog(ff1.T, strain[:, 2] / strain[:, 0],
               label      = 'Cleaned/DARM',
               c          = 'xkcd:slate',
               rasterized = True)

    ax2.loglog(ff1.T, [1 for _ in range(len(ff1.T))],
               label      = 'No Subtraction',
               c          = 'xkcd:black',
               linestyle  = ':',
               rasterized = True)

    ax2.grid(True, which='minor')
    ax2.set_ylabel(r'PSD Ratio')
    ax2.set_ylim(top=5.0)
    ax2.legend(fontsize='x-small', loc=1)

    plt.subplots_adjust(hspace=0.075)

    # save figure,
    if plotDir.endswith('/'):
        plotDir = ''.join(plotDir[:-1])

    plt.savefig('{0}/{1}.png'.format(plotDir, saveas))

    if savepdf:
        plt.savefig('{0}/{1}.pdf'.format(plotDir, saveas))


def plot_progress(darm, predictions,
                  fs      = 512,
                  nfft    = 4096,
                  loops   = ['Loop_0', 'Loop_1', 'Loop_2'],
                  plotDir = 'Plots',
                  title   = 'Total Network Subtraction',
                  saveas  = 'total_subtraction',
                  savepdf = False,
                  fmin    = 4,
                  fmax    = 256):
    """
    plot the psd of the DARM (testing) timeseries, the prediction timeseries
    calculated by the network, and the residual.

    Parameters
    ----------
    target : `numpy.ndarray`
        validation DARM timeseries
    prediction: `numpy.ndarray`
        validation prediction timeseries
    fs : `int`
        data sample rate
    nfft : `int`
        overlapping windows
    loops : `list`
        list of loop iterations trained and tested on
    plotDir : `str`
        path to store plots
    title : `str`
        plot title
    saveas : `str`
        output filename for plot
    savepdf : `bool`
        when set to True, both png and pdf filetypes will be saved
    fmin : `float`
        x-axis min
    fmax : `float`
        x-axis max
    """
    residual = np.copy(darm)
    darm = darm.reshape(len(darm), 1)
    predictions = np.array(predictions).T

    for i in range(predictions.shape[1]):
        residual -= predictions[:, i]

    residual = residual.reshape(len(residual), 1)
    targets = np.hstack((darm, predictions, residual)).T
    ff1, pp = sig.welch(targets,
                        fs      = fs,
                        nperseg = nfft,
                        axis    = -1)

    strain = np.sqrt(pp).T

    fig, (ax1, ax2) = plt.subplots(nrows       = 2,
                                   sharex      = True,
                                   figsize     = (8, 6),
                                   gridspec_kw = {'height_ratios': [2, 1]})

    ax1.set_prop_cycle(cycler('color', ['xkcd:black',
                                        'xkcd:twilight blue',
                                        'xkcd:pumpkin',
                                        'xkcd:jade',
                                        'xkcd:cement',
                                        'xkcd:forest green',
                                        'xkcd:light eggplant']))

    labels = ['DARM'] + loops + ['Cleaned']
    linestyles = ['-']
    for i in range(len(loops)):
        linestyles.append(':')
    linestyles.append('-')

    for i in range(len(labels)):
        ax1.loglog(ff1.T, strain[:,i],
                   alpha      = 0.8,
                   rasterized = True,
                   linestyle  = linestyles[i],
                   label      = labels[i])

    ax1.legend(fontsize='small', loc=1)
    ax1.set_xlim([fmin, fmax])
    # ax1.set_ylim([1e-28, 5e-18])
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
    ax1.set_title(title)

    # Plot the ratio of the PSDs. Max ratio where DARM = subtracted
    ax2.loglog(ff1.T, strain[:, -1] / strain[:, 0],
               label      = 'Cleaned/DARM',
               c          = 'xkcd:slate',
               rasterized = True)

    ax2.loglog(ff1.T, [1 for _ in range(len(ff1.T))],
               label      = 'No Subtraction',
               c          = 'xkcd:black',
               linestyle  = ':',
               rasterized = True)

    ax2.grid(True, which='minor')
    ax2.set_ylim(top=5)
    ax2.set_ylabel(r'Final PSD Ratio')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.legend(fontsize='x-small', loc='upper right')

    plt.subplots_adjust(hspace=0.075)

    # save plots
    if plotDir.endswith('/'):
        plotDir = ''.join(plotDir[:-1])

    plt.savefig('{0}/{1}.png'.format(plotDir, saveas))

    if savepdf:
        plt.savefig('{0}/{1}.pdf'.format(plotDir, saveas))

    plt.close()


def split_dict(d, elements):
    """
    split_dict is used in plot_density in order to take a dict `d` of
    length `n` and split it into a list of dicts of length `elements`

    Parameters
    ----------
    d : `dict`
        input dictionary

    elements : `int`
        number of elements in each sub-dict

    Returns
    -------
    output : `list`
        list containing dicts of length `elements`
    """
    output = []
    chunks = int(len(d) / elements) + int(bool(len(d) % elements))
    for i in range(chunks):
        temp = {}
        if i < chunks - 1:
            for j in range(elements):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]
        else:
            remainder = len(d) - elements * i
            for j in range(remainder):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]

        output.append(temp)

    return output


def plot_channel_correlations(datafile,
                              plotNum   = 4,
                              data_type = None,
                              seconds   = 15,
                              plotDir   = '.'):
    """
    plot_channel_correlations calculates comparisons between channels
    in order to show which channels may contain 'features' of DARM.

    Parameters
    ----------
    datafile : `str`
        mat file to analyze

    plotNum : `int`
        Number of plots per image

    data_type : `str`
        use either 'real' or 'mock' data

    seconds : `int`
        How many seconds of data to query. NOTE: using times longer
        than ~30 seconds start to take a really long time to compute.
        If possible, use <= 30 seconds unless you're particularly
        patient :)

    plotDir : `str`
        path to store plots
    """
    # Get the datafile and type
    if data_type == None:
        if "data_array" in datafile:
            data_type = "real"
        elif "DARM_with" in datafile:
            data_type = "mock"

    # Get the data and separate darm from witnesses
    dataset, fs = ppr.get_dataset(datafile, data_type=data_type)

    # Scale it for nicer plotting
    scaler  = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    # Cut it down to a reasonable size (1 minute)
    duration = int(fs * seconds)
    temp     = np.zeros(shape=(duration, dataset.shape[1]))

    for i in range(dataset.shape[1]):
        temp[:, i] = dataset[:duration, i]

    darm    = temp[:, 0]
    witness = temp[:, 2:]

    # Collect the channels or make them up
    if data_type == 'real':
        chans = sio.loadmat(datafile)['chans']
        chans = [c for c in chans if not "DARM" in c]
    else:
        chans = []
        for i in range(witness.shape[1]):
            chans.append("Witness {}".format(i + 1))

    data = dict(zip(chans, witness.T))
    data = split_dict(data, plotNum - 1)  # Not including DARM

    # Make sure the output directory exists
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    for index, chunk in enumerate(data):
        sys.stdout.write('\rMaking plot {0}/{1}'.format(index + 1, len(data)))
        sys.stdout.flush()
        if index == len(data) - 1:
            print('')

        # Add DARM and convert to DataFrame object
        chunk["DARM"] = darm
        df = pd.DataFrame(chunk)

        # Make the plots
        G = sns.PairGrid(df)

        G.map_diag(sns.distplot,
                   fit   = stats.gamma,
                   kde   = False,
                   rug   = False,
                   color = 'k')

        cmap = sns.cubehelix_palette(as_cmap = True,
                                     dark    = 0,
                                     light   = 1,
                                     reverse = True)
        G.map_offdiag(sns.kdeplot,
                      cmap         = cmap,
                      shade        = True,
                      shade_lowest = False)

        plt.suptitle('{0} Data Channel Correlations'.format(data_type.title()))
        plt.savefig('{0}/{1}_channel_corr_{2}.png'.format(plotDir, data_type, index))
        plt.savefig('{0}/{1}_channel_corr_{2}.pdf'.format(plotDir, data_type, index))
        plt.close()
