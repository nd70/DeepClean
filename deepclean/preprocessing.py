
try:
    import nds2
except:
    nds2 = None
import os
os.environ['NDS2_CLIENT_ALLOW_DATA_ON_TAPE'] = '1'
import pandas as pd
import scipy.signal as sig
import scipy.io as sio
import sys
import numpy as np

from astropy.time import Time
from collections import OrderedDict
import configparser

from sklearn.preprocessing import StandardScaler, MinMaxScaler

DEFAULTS = {
    'Data': {
        'chanlist': 'deepclean/ChanList_H1.txt',
        'datafile': 'Data/H1_data_August.mat',
        'data_type': 'real',
        'data_start': '2017-08-14 02:00:00',
        'duration': '2048',
        'fs': '512',
        'ifo': 'H1',
        'output': 'None',
        'portNumber' : '31200',
        'save_mat': 'True',
        },
    'Webpage': {
        'basedir': 'DeepClean/',
        },
    'Run': {
        'recurrent_activation': 'tanh',
        'recurrent_initializer': 'glorot_uniform',
        'dense_activation': 'relu',
        'dense_bias_initializer': 'zeros',
        'dense_kernel_initializer': 'glorot_uniform',
        'beta_1': '0.9',
        'beta_2': '0.999',
        'decay': 'None',
        'dropout': '0.1',
        'epochs': '1',
        'epsilon': '1e-8',
        'fmin': '3',
        'fmax': '20',
        'hc_offset': '0',
        'highcut': '9.0',
        'lookback': '15',
        'loss': 'psd',
        'lowcut': '3.0',
        'lr': 'None',
        'momentum': '0.0',
        'nesterov': 'False',
        'N_bp': '8',
        'optimizer': 'adam',
        'plotDir': 'Plots',
        'postFilter': 'True',
        'preFilter': 'True',
        'recurrent_dropout': '0.0',
        'rho': 'None',
        'subsystems': 'all',
        'tfrac': '0.5',
        'verbose': '1',
        'logDir': 'log',
        'tuningDir': 'Tuning',
        }
    }


def lstm_lookback(data, n_in=1, n_out=1):
    """
    create lookback in the dataset

    Parameters
    ----------
    data : `numpy.ndarray`
        dataset for training and testing
    n_in : `int`
        number of timesteps to lookback
    n_out : `int`
        number of timesteps to forecast

    Returns
    -------
    combined : `numpy.ndarray`
        dataset with lookback
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # predict future timestep
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    combined = pd.concat(cols, axis=1)
    combined.columns = names
    combined.dropna(inplace=True)

    return combined


def phase_filter(dataset,
                 lowcut  = 4.0,
                 highcut = 20.0,
                 order   = 8,
                 btype   = 'bandpass',
                 fs      = 512):
    """
    phase preserving bandpass filter

    Parameters
    ----------
    btype : `str`
        filter type
    dataset : `numpy.ndarray`
        dataset for training and testing
    fs : `int`
        data sample rate
    highcut : `float`
        stop frequency for filter
    lowcut : `float`
        start frequency for filter
    order : `int`
        bandpass filter order

    Returns
    -------
    dataset : `numpy.ndarray`
        bandpassed dataset for training and testing
    """

    # Normalize the frequencies
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq

    # Make and apply filter
    z, p, k = sig.butter(order, [low, high], btype=btype, output='zpk')
    sos = sig.zpk2sos(z, p, k)

    if dataset.ndim == 2:
        for i in range(dataset.shape[1]):
            dataset[:, i] = sig.sosfiltfilt(sos, dataset[:, i])
    else:
        dataset = sig.sosfiltfilt(sos, dataset)

    return dataset


def get_run_params(ini_file, section):
    """
    function for reading parameters from the config file

    Parameters
    ----------
    ini_file : `str`
        path to config file
    section : `str`
        config file section to read from

    Returns
    -------
    run_params : `dict`
        dict of params from supplied config file and section
    """
    if section in ('Data', 'Webpage'):
        settings = configparser.ConfigParser(DEFAULTS[section])
    elif 'Loop' in section:
        settings = configparser.ConfigParser(DEFAULTS['Run'])
    else:
        settings = configparser.ConfigParser()

    # ConfigParser cannot read some default values if this is enabled
    # settings.optionxform=str

    settings.read(ini_file)
    run_params = OrderedDict()

    if section == 'Data':
        run_params['datafile'] = settings.get(section, 'datafile')
        run_params['data_type'] = settings.get(section, 'data_type')
        run_params['save_mat'] = settings.getboolean(section, 'save_mat')
    elif section == 'To_Run':
        run_dict = settings.items('To_Run')
        for i, sect in enumerate(run_dict):
            loop = 'Loop_{}'.format(i)
            run_params[loop] = settings.getboolean(section, loop)

    elif section == 'Webpage':
        run_params['basedir'] = settings.get(section, 'basedir')

    else:
        # network setting
        run_params['recurrent_activation'] = settings.get(section, 'recurrent_activation')
        run_params['recurrent_initializer'] = settings.get(section, 'recurrent_initializer')
        run_params['dense_activation'] = settings.get(section, 'dense_activation')
        run_params['dense_kernel_initializer'] = settings.get(
            section, 'dense_kernel_initializer')
        run_params['dense_bias_initializer'] = settings.get(section, 'dense_bias_initializer')
        run_params['dropout'] = settings.getfloat(section, 'dropout')
        run_params['recurrent_dropout'] = settings.getfloat(
            section, 'recurrent_dropout')
        run_params['rho'] = settings.get(section, 'rho')
        if run_params['rho'] == 'None':
            run_params['rho'] = None
        else:
            run_params['rho'] = settings.getfloat(section, 'rho')

        # loss setting
        run_params['loss'] = settings.get(section, 'loss')

        # training setting
        run_params['epochs'] = settings.getint(section,   'epochs')
        run_params['optimizer'] = settings.get(section, 'optimizer')
        run_params['beta_1'] = settings.getfloat(section, 'beta_1')
        run_params['beta_2'] = settings.getfloat(section, 'beta_2')
        run_params['epsilon'] = settings.getfloat(section, 'epsilon')
        run_params['decay'] = settings.get(section, 'decay')
        if run_params['decay'] == 'None':
            run_params['decay'] = None
        else:
            run_params['decay'] = settings.getfloat(section, 'decay')
        run_params['momentum'] = settings.getfloat(section, 'momentum')
        run_params['nesterov'] = settings.getboolean(section, 'nesterov')
        run_params['lr'] = settings.get(section, 'lr')
        if run_params['lr'] == 'None':
            run_params['lr'] = None
        else:
            run_params['lr'] = settings.getfloat(section, 'lr')
        run_params['tfrac'] = settings.getfloat(section, 'tfrac')
        run_params['verbose'] = settings.getint(section, 'verbose')

        # filter settings
        run_params['lowcut'] = settings.getfloat(section, 'lowcut')
        run_params['highcut'] = settings.getfloat(section, 'highcut')
        run_params['N_bp'] = settings.getint(section, 'N_bp')
        run_params['hc_offset'] = settings.getfloat(section, 'hc_offset')

        # preprocessing setting
        run_params['lookback'] = settings.getint(section, 'lookback')
        run_params['postFilter'] = settings.getboolean(section, 'postFilter')
        run_params['preFilter'] = settings.getboolean(section, 'preFilter')

        # plotting setting
        run_params['fmax'] = settings.getfloat(section, 'fmax')
        run_params['fmin'] = settings.getfloat(section, 'fmin')
        run_params['plotDir'] = settings.get(section, 'plotDir')

        subsystems = settings.get(section, 'subsystems')
        if ',' in subsystems:
            subsystems = [s.strip() for s in subsystems.split(',')]
        else:
            subsystems = [subsystems]
        run_params['subsystems'] = subsystems
        run_params['logDir'] = settings.get(section, 'logDir')

    return run_params


def get_dataset(datafile, subsystems='all', data_type='real', chanlist='all',
                return_chans=False):
    """
    get_dataset reads in a datafile and returns the dataset
    used during training. Optionally, particular subsystems
    may be given as witness channels

    Parameters
    ----------
    datafile : `string`
        full path to mat file

    subsystems : `list`
        subsystems to include in dataset
        e.g. subsystems = ['ASC', 'CAL', 'HPI', 'SUS']

    data_type : `str`
        use either "real", "mock" or "scatter"

    return_chans: `bool`
        if True, return an array of channel name

    Returns
    -------
    dataset : `numpy.ndarray`
        test data. includes all channels except darm

    fs : `int`
        sample rate of data
    """
    mat_file = sio.loadmat(datafile)

    if data_type == 'mock' or data_type == 'scatter':
        darm = mat_file['darm'].T
        wits = mat_file['wit'].T
        fs   = mat_file['fs'][0][0]
        chans_list = None

    elif data_type == 'real':
        chans = [str(c.strip()) for c in mat_file['chans']]
        data  = mat_file['data']
        darm  = data[0, :].T
        fs    = mat_file['fsample']
        chans_list = [chans[0]]

        if isinstance(subsystems, str):
            subsystems = [subsystems]

        if subsystems[0].lower() == 'all':
            if not chanlist == 'all':
                data_dict = dict(list(zip(chans, data)))
                for k, v, in list(data_dict.items()):
                    if k in chanlist:
                        chans_list.append(k)
                        wits.append(v)
                wits = np.array(wits).T
            else:
                chans_list = chans
                wits = data.T[:, 1:]
        else:
            wits = []
            data_dict = dict(list(zip(chans, data)))
            for subsystem in subsystems:
                for k, v, in list(data_dict.items()):
                    if not chanlist == 'all':
                        if not k in chanlist: continue
                    if subsystem.upper() in k:
                        chans_list.append(k)
                        wits.append(v)
            wits = np.array(wits).T

    dataset = np.zeros(shape=(darm.shape[0], wits.shape[1] + 1))
    dataset[:, 0]  = np.squeeze(darm)
    dataset[:, 1:] = wits

    if return_chans:
        return dataset, fs, chans_list
    return dataset, fs


def read_chans_and_times(chanlist):
    """
    Allows user to apply custom times to each supplied witness
    channel. If no time is specified, the default time is used.

    Parameters
    ----------
    chanlist : `str`
        text file containing the target (first channel given) and
        the witnesses channels

    Returns
    -------
    chans : `list`
        list of target and witness channels
    times : `list`
        list of times to query for each of the requested channels
    """
    chans_times = []
    with open(chanlist) as f:
        for line in f.readlines():
            if ',' in line:
                terms = line.split(',')
                chans = terms[0].strip()
                times = int(terms[-1].strip('\n').strip())
            else:
                chans = line.strip('\n')
                times = None

            chans_times.append((chans, times))

    chans = [chans_times[i][0] for i in range(len(chans_times))]
    times = [chans_times[i][1] for i in range(len(chans_times))]
    return chans, times


def stream_data(ini_file):
    """
    Stream the requested data from nds2.

    Parameters
    ----------
    ini_file : `str`
        path to configuration file which contains the pipeline parameters

    Returns
    -------
    vdata : `ndarray`
        numpy array containing the requested channel data. if `save_mat` is True,
        the data is saved to a mat file and not returned
    fsup : `int`
        sample rate of the collected data
    """

    # Read config file
    try:
        settings = ConfigParser.ConfigParser()
    except:
        settings = configparser.ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)


    # Unpack configs
    dur        = settings.getint('Data', 'duration')
    fname      = settings.get('Data', 'chanlist')
    fs         = settings.getint('Data', 'fs')
    ifo        = settings.get('Data', 'ifo')
    output     = settings.get('Data', 'output')
    portNumber = settings.getint('Data', 'portNumber')
    save       = settings.getboolean('Data', 'save_mat')
    data_dir   = settings.get('Data', 'data_dir')
    times      = settings.get('Data', 'data_start')

    nds_osx = ('/opt/local/Library/Frameworks/Python.framework/' +
               'Versions/2.7/lib/python2.7/site-packages/')
    nds_sandbox = '/usr/lib/python2.7/dist-packages/'

    import sys
    if os.path.exists(nds_osx):
        sys.path.append(nds_osx)
    elif os.path.exists(nds_sandbox):
        sys.path.append(nds_sandbox)

    # Collect channels and times
    chan_head = ifo + ':'
    chanlines, custom_times = read_chans_and_times(fname)
    channels = [chan_head + line for line in chanlines]

    # Get data start time
    if ifo == 'L1':
        ndsServer = 'nds.ligo-la.caltech.edu'
    elif ifo == 'H1':
        ndsServer = 'nds.ligo-wa.caltech.edu'
    else:
        sys.exit("unknown IFO specified")

    # Setup connection to the NDS
    try:
        conn = nds2.connection(ndsServer, portNumber)
    except RuntimeError:
        print('ERROR: Need to run `kinit albert.einstein` before nds2 '
              'can establish a connection')
        sys.exit(1)

    #if __debug__:
    #    print(("Output sample rate: {} Hz".format(fsup)))
    #    # print("Channel List:\n-------------")
    #    # print("\n".join(channels))

    # Setup start and stop times
    t = Time(times, format='iso', scale='utc')
    t_start = int(t.gps)

    print(("Getting data from " + ndsServer + "..."))
    data = []
    for i in range(len(custom_times)):
        if custom_times[i] == None:
            custom_times[i] = t_start

        try:
            temp = conn.fetch(custom_times[i], custom_times[i] + dur, [channels[i]])
            sys.stdout.write("\033[0;32m")
            sys.stdout.write('\r  [{}] '.format('\u2713'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(channels[i]))
            sys.stdout.write('\n')
            sys.stdout.flush()
        except:
            sys.stdout.write("\033[1;31m")
            sys.stdout.write('\r  [{}] '.format('\u2717'))
            sys.stdout.write("\033[0;0m")
            sys.stdout.write('{} '.format(channels[i]))
            sys.stdout.write('\n')
            sys.stdout.flush()

        data.append(temp)

    # Get the data and stack it (the data are the columns)
    vdata = []
    for k in range(len(channels)):
        fs_data = data[k][0].channel.sample_rate
        resample_data = sig.resample(data[k][0].data,
                                     int(1.*len(data[k][0].data)*fs/fs_data))
        vdata.append(resample_data)

    if save:
        if data_dir == "None":
            data_dir = 'Data'

        if not os.path.isdir(data_dir):
            os.system('mkdir %s' % data_dir)

        # save to a hdf5 format
        if output == "None":
            funame = os.path.join(data_dir, '%s_%d_%d.mat' % (ifo, t_start, dur))
        else:
            funame = os.path.join(data_dir, output)

        sio.savemat(funame,
                    mdict={'data': vdata, 'fsample': fs, 'chans': channels},
                    do_compression=True)

        print(("Data saved as " + funame))
    else:
        return np.array(vdata).T, fs


# define lookback function
def do_lookback(data, steps=1, validation=False):
    """
    modify the dataset to include previous timesteps to feed into the
    recurrent network. this helps to captiure long-term dependencies

    Parameters
    ----------
    data : `ndarray`
        numpy array containing the initial dataset
    steps : `int`
        number of previous time-steps to include
    validation : `bool`
        used for handling 1D arrays

    Returns
    -------
    temp : `ndarray`
        data array with added (previous) time-steps. if original array
        was M x N and L steps of lookback were requested, then the output
        is of shape M x L x N

    """
    temp = np.zeros((data.shape[0] - steps, steps + 1, data.shape[1]))
    temp[:, 0, :] = data[steps:, :]
    for i in range(temp.shape[0]):
        temp[i, 1:] = data[i:i + steps][::-1]

    if validation:
        temp = temp.reshape((temp.shape[0], temp.shape[1]))

    for i in range(temp.shape[0]):
        temp[i, :] = temp[i, :][::-1]

    return temp


def get_list(settings, section, key, dtype):
    """
    convienient function to read list parameters from the config file

    Parameters
    ----------
    settings : ConfigParser
        ConfigParser object
    section : `str`
        config file section to read from
    key : `str`
        section key to read from
    dtype : type
        data type

    Returns
    -------
    val : dtype or list of dtype
        value of section key
    """
    val = settings.get(section, key)
    if ',' in val:
        val = [v.strip() for v in val.split(',')]
        if dtype == str:
            return val
        return list(map(dtype, val))
    return dtype(val)


def get_tuning_params(ini_file, section):
    """
    function for reading tuning parameters from the config file

    Parameters
    ----------
    ini_file : `str`
        path to config file
    section : `str`
        config file section to read from

    Returns
    -------
    run_params : `dict`
        dict of params from supplied config file and section
    """

    settings = configParser.ConfigParser(DEFAULTS['Run'])

    # ConfigParser cannot read some default values if this is enable
    # settings.optionxform = str

    settings.read(ini_file)
    run_params = OrderedDict()

    # Hyperparameters not for tuning
    run_params['epochs'] = settings.getint(section, 'epochs')
    run_params['lowcut'] = settings.getfloat(section, 'lowcut')
    run_params['highcut'] = settings.getfloat(section, 'highcut')
    run_params['lookback'] = settings.getint(section, 'lookback')
    run_params['loss'] = settings.get(section, 'loss')
    run_params['postFilter'] = settings.getboolean(section, 'postFilter')
    run_params['preFilter'] = settings.getboolean(section, 'preFilter')
    run_params['N_bp'] = settings.getint(section, 'N_bp')
    subsystems = settings.get(section, 'subsystems')
    if ',' in subsystems:
        subsystems = [s.strip() for s in subsystems.split(',')]
    else:
        subsystems = [subsystems]
    run_params['subsystems'] = subsystems
    run_params['tfrac'] = settings.getfloat(section, 'tfrac')
    run_params['tuningDir'] = settings.get(section, 'tuningDir')

    run_params['nesterov'] = settings.getboolean(section, 'nesterov')


    # Hyperparameters for tuning
    run_params['recurrent_act'] = get_list(
        settings, section, 'recurrent_act', str)
    run_params['recurrent_init'] = get_list(
        settings, section, 'recurrent_init', str)
    run_params['dense_act'] = get_list(
        settings, section, 'dense_act', str)
    run_params['dense_kernel_init'] = get_list(
        settings, section, 'dense_kernel_init', str)
    run_params['dense_bias_init'] = get_list(
        settings, section, 'dense_bias_init', str)
    run_params['dropout'] = get_list(settings, section, 'dropout', float)
    run_params['recurrent_dropout'] = get_list(
        settings, section, 'recurrent_dropout', float)
    run_params['rho'] = settings.get(section, 'rho')
    if run_params['rho'] == 'None':
        run_params['rho'] = None
    else:
        run_params['rho'] = settings.getfloat(section, 'rho')


    run_params['optimizer'] = get_list(settings, section, 'optimizer', str)
    run_params['beta_1'] = get_list(settings, section, 'beta_1', float)
    run_params['beta_2'] = get_list(settings, section, 'beta_2', float)
    run_params['decay'] = settings.get(section, 'decay')
    if run_params['decay'] == 'None':
        run_params['decay'] = None
    else:
        run_params['decay'] = get_list(settings, section, 'decay', float)
    run_params['dropout'] = get_list(settings, section, 'dropout', float)
    run_params['epsilon'] = get_list(settings, section, 'epsilon', float)
    run_params['lr'] = settings.get(section, 'lr')
    if run_params['lr'] == 'None':
        run_params['lr'] = None
    else:
        run_params['lr'] = get_list(settings, section, 'lr', float)
    run_params['momentum'] = get_list(settings, section, 'momentum', float)
    run_params['recurrent_dropout'] = get_list(
        settings, section, 'recurrent_dropout', float)
    run_params['rho'] = settings.get(section, 'rho')
    if run_params['rho'] == 'None':
        run_params['rho'] = None
    else:
        run_params['rho'] = get_list(settings, section, 'rho', float)

    return run_params

# convienient function
def load_data(
    datafile   = 'Data/H1_data_array.mat',
    data_type  = 'real',
    ini_file   = 'configs/configs.ini',
    lowcut     = 3.0,
    highcut    = 60.0,
    N_bp       = 8,
    preFilter  = True,
    save_mat   = False,
    clean_darm = [],
    lookback   = 15,
    subsystems = 'all',
    return_chans = False,
    **kwargs
    ):

    # Read config file
    try:
        settings = ConfigParser.ConfigParser()
    except:
        settings = configparser.ConfigParser()
    settings.optionxform=str
    settings.read(ini_file)

    # Unpack configs
    dur        = settings.getint('Data', 'duration')
    fname      = settings.get('Data', 'chanlist')
    fs         = settings.getint('Data', 'fs')
    ifo        = settings.get('Data', 'ifo')
    output     = settings.get('Data', 'output')
    portNumber = settings.getint('Data', 'portNumber')
    save       = settings.getboolean('Data', 'save_mat')
    data_dir   = settings.get('Data', 'data_dir')
    times      = settings.get('Data', 'data_start')

    # Setup start and stop times
    t = Time(times, format='iso', scale='utc')
    t_start = int(t.gps)

    if save:
        if data_dir == "None":
            data_dir = 'Data'

        # save to a hdf5 format
        if output == "None":
            datafile = os.path.join(data_dir, '%s_%d_%d.mat' % (ifo, t_start, dur))
        else:
            datafile = os.path.join(data_dir, output)

    # load dataset and scale
    if save:
        if not os.path.isfile(datafile):
            stream_data(ini_file)
        data = get_dataset(datafile,
                           data_type  = data_type,
                           subsystems = subsystems,
                           return_chans = return_chans)
        if return_chans:
            dataset, fs, chans = data
        else:
            dataset, fs = data
    else:
        dataset, fs = stream_data(ini_file)
    fs = int(fs)

    # keep raw, unscaled data for testing
    nd = np.copy(dataset)

    # feed in previously cleaned results into testing sample
    if len(clean_darm) > 0:
        dataset[-len(clean_darm):, 0] = clean_darm

    # bandpass filter
    if preFilter:
        dataset = phase_filter(dataset,
                               fs      = fs,
                               order   = N_bp,
                               lowcut  = lowcut,
                               highcut = highcut)

    # normalize and standardize
    std_scaler = StandardScaler()
    scaled         = std_scaler.fit_transform(dataset)

    # return scaled data, original dataset, scaler, and fs
    scaler = std_scaler

    if return_chans and save_mat:
        return scaled, nd, fs, scaler, chans
    return scaled, nd, fs, scaler

