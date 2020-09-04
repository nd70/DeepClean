

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def dense_bn(
    x,
    n,
    activation=tf.nn.relu,
    kernel_initializer=None,
    bias_initializer=None,
    dropout=0.0,
    batch_norm=False,
    is_train=False,
    name='dense_bn'):

    with tf.name_scope(name):
        dense = tf.layers.Dense(n, activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        x = dense.apply(x)

        if batch_norm:
            x = tf.layers.batch_normalization(x, training=is_train)

        if dropout > 0.0:
            x = tf.layers.dropout(x, rate=dropout, training=is_train)

    return x


class DeepClean(object):

    def fit(self, x_train, y_train, sess=None, epochs=1, batch_size=32,
        validation_data=None, shuffle=True, verbose=True, logdir=None):
        """ Train model for time series regression and return train/val loss """

        N, T, D = x_train.shape

        # set up tensorboard
        train_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('Tensorboard/{}/train'.format(logdir), sess.graph)
        val_summary = tf.Summary()
        val_summary.value.add(tag='loss', simple_value=None)
        val_writer = tf.summary.FileWriter('Tensorboard/{}/test'.format(logdir))

        if validation_data is not None:
            x_val, y_val = validation_data

        train, val = [], []
        i_running = 0
        for ep in range(epochs):

            # get all starting indices
            indices = np.arange(0, N, batch_size)
            if shuffle:
                np.random.shuffle(indices)

            train_loss = 0
            for i, start in enumerate(indices):

                # get training batch
                start = np.random.randint(N-batch_size)
                end = start + batch_size
                x_train_batch = x_train[start:end].reshape(-1, D, T)
                y_train_batch = y_train[start:end].reshape(-1, 1)

                feed_dict = {
                self.input: x_train_batch,
                self.labels: y_train_batch,
                self.is_train: True,
                self.params['loss']['nfft']: min(4096, int(x_train_batch.shape[0]//16))}

                # train and return batch loss
                if i % 100 != 0 or logdir is None:
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                else:
                    _, loss, s = sess.run(
                        [self.optimizer, self.loss, train_summary], feed_dict=feed_dict)
                    train_writer.add_summary(s, i_running) # tensorboard

                train_loss += loss
                if i % 100 == 0 and verbose:
                    print('- epoch %3d - iter %5d - loss: %.4e' % (ep, i, loss))

                # update running index
                i_running += 1

            # average over the size of the training set
            train_loss /= N
            train.append(train_loss)

            # evaluate val set at the end of each epoch
            if validation_data is not None:
                val_loss = self.evaluate(x_val, y_val, sess, batch_size,)
                val.append(val_loss)

                # tensorboard
                val_summary.value[0].simple_value = val_loss
                val_writer.add_summary(val_summary, i_running)

                if verbose:
                    print('- epoch %3d - loss %.4e - val_loss: %.4e' % (ep, train_loss, val_loss))
            elif verbose:
                    print('- epoch %3d - loss %.4e' % (ep, train_loss))

        # return stats
        if validation_data is not None:
            return train, val
        return train


    def evaluate(self, x, y, sess=None, batch_size=None, verbose=True):
        """ Return loss in test mode"""

        # get shape
        N, T, D = x.shape
        avg_loss = 0
        for i in range(N//batch_size+1):

            # get batch
            start = i*batch_size
            end = start + batch_size
            x_batch = x[start:end].reshape(-1, D, T)
            y_batch = y[start:end].reshape(-1, 1)

            # evaluate
            feed_dict={
            self.input: x_batch,
            self.labels: y_batch,
            self.is_train: False,
            self.params['loss']['nfft']: min(4096, int(x_batch.shape[0]//16))}

            loss = sess.run(self.loss, feed_dict=feed_dict)
            avg_loss += loss

        avg_loss /= N
        return avg_loss

    def predict(self, x, sess=None, batch_size=32, verbose=True):
        """ Generates output predictions """

        # get shape
        N, T, D = x.shape
        predictions = []
        for i in range(N//batch_size+1):

            # get batch
            start = i*batch_size
            end = start + batch_size
            x_batch = x[start:end].reshape(-1, D, T)

            # predict
            pred = sess.run(self.predictions,
                feed_dict={self.input: x_batch, self.is_train: False})
            predictions.append(pred)

        # convert list into numpy array
        predictions = np.concatenate(predictions)
        return predictions

    def _build_graph(
        self,
        x,
        recurrent_activation='tanh',
        dense_activation='relu',
        recurrent_initializer='glorot_uniform',
        dense_kernel_initializer='glorot_uniform',
        dense_bias_initializer='zeros',
        dropout=0.1):

        # initializer and activation
        initializers = {
            'glorot_uniform': tf.glorot_uniform_initializer(),
            'glorot_normal': tf.glorot_normal_initializer(),
            'zeros': tf.zeros_initializer(),
            'ones': tf.ones_initializer(),
            }
        recurrent_initializer = initializers[recurrent_initializer]
        dense_kernel_initializer  = initializers[dense_kernel_initializer]
        dense_bias_initializer = initializers[dense_bias_initializer]

        activations = {
            'tanh': tf.nn.tanh,
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
        }
        recurrent_activation = activations[recurrent_activation]
        dense_activation = activations[dense_activation]

        # features extraction using lstm
        with tf.name_scope('features'):
            multi_lstm = rnn.MultiRNNCell([
                rnn.LSTMCell(
                    16,
                    activation=recurrent_activation,
                    initializer=recurrent_initializer,
                    name='layer0'),
                rnn.LSTMCell(
                    8,
                    activation=recurrent_activation,
                    initializer=recurrent_initializer,
                    name='layer1')])
            x, state = rnn.static_rnn(multi_lstm, x, dtype=tf.float32)
            x = x[-1]
            self.layers.append(x)

        # predict from features
        with tf.name_scope('predict'):
            x = dense_bn(x, 32,
                activation=dense_activation,
                kernel_initializer=dense_kernel_initializer,
                bias_initializer=dense_bias_initializer,
                dropout=dropout,
                batch_norm=True,
                is_train=self.is_train,
                name='layer0')
            self.layers.append(x)

            x = dense_bn(x, 16,
                activation=dense_activation,
                kernel_initializer=dense_kernel_initializer,
                bias_initializer=dense_bias_initializer,
                dropout=dropout,
                batch_norm=True,
                is_train=self.is_train,
                name='layer1')
            self.layers.append(x)

        x = tf.layers.dense(x, 1, name='layer2')
        return x


    def _loss(self, loss_type='psd', nfft=None, lowcut=3.0, highcut=60.0, fs=512, psd_scale=1):

        labels = self.labels
        predictions = self.predictions

        if loss_type == 'psd':

            # calculate the residual
            res = labels - predictions
            n = tf.shape(res)[0]
            res = tf.reshape(res, (n, ))

            # Define PSD paramaters
            nperseg = nfft  # length of each segment
            nfreq = nperseg//2 + 1  # length of the psd
            dn = nperseg//2  # step
            nseg = n//dn-1   # number of iterations

            # calculate the PSD from rFFT
            res_psd = tf.zeros(nfreq, dtype=tf.float32)
            window = tf.contrib.signal.hann_window(nperseg, dtype=tf.float32)
            res *= psd_scale

            i = tf.constant(0)
            while_condition =  lambda i, res_psd: tf.less(i, nseg)
            def while_body(i, res_psd):
                min_ii = i*dn
                max_ii = min_ii + nperseg
                res_psd += tf.abs(tf.spectral.rfft(res[min_ii:max_ii]*window))**2

                # increment
                i = tf.add(i, 1)
                return i, res_psd
            i, res_psd = tf.while_loop(while_condition, while_body, [i, res_psd])
            res_psd = tf.divide(res_psd, tf.cast(nseg, dtype=tf.float32))

            # apply frequency cut
            freq = tf.lin_space(0., fs/2., nfreq)
            mask = tf.where(
                tf.logical_and(lowcut <= freq, freq <= highcut),
                tf.ones(tf.shape(freq)),
                tf.zeros(tf.shape(freq)))
            res_psd = tf.multiply(res_psd, mask)

            # calculate sum
            loss = tf.reduce_sum(res_psd)

        elif loss_type == 'mse':
            # mean squared error
            loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

        elif loss_type == 'mae':
            # mean absolute error
            loss = tf.losses.mean_absolute_error(labels=labels, predictions=predictions)
        else:
            print('ERROR: Unknown loss')
            sys.exit(1)

        return loss

    def _optimizer(
        self,
        optimizer=None,
        lr=None,
        momentum=0.0,
        nesterov=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        rho=None,
        decay=None,):

        opt = optimizer.lower()

        # get learning rate
        lr_defaults = {
            'sgd' : 1e-2,
            'adam': 1e-3,
            'adamax': 2e-3,
            'nadam': 2e-3,
            'rmsprop': 1e-3,
            'adagrad': 1e-2,
            'adadelta': 1.0,
        }
        learning_rate = lr_defaults[opt] if lr is None else lr

        # learning rate decay
        if decay is None:
            decay = 0.0
            if opt == 'nadam':
                decay = 0.04
        global_step = tf.Variable(0, dtype=tf.int32)
        learning_rate = learning_rate/(1 + tf.pow(
            decay, tf.cast(global_step, tf.float32)))

        if opt == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
                                                   momentum = momentum,
                                                   use_nesterov = nesterov)
        elif opt == 'adam' or opt == 'adamax' or opt == 'nadam':
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                               beta1         = beta_1,
                                               beta2         = beta_2,
                                               epsilon       = epsilon)
        elif opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate     = learning_rate,
                                                  decay             = decay,
                                                  momentum          = momentum,
                                                  epsilon           = epsilon)
        elif opt == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate)

        elif opt == 'adadelta':
            rho = 0.95 if rho is None else rho
            optimizer = tf.train.AdadeltaOptimizer(learning_rate    = learning_rate,
                                                   rho              = rho,
                                                   epsilon          = epsilon)
        else:
            print('ERROR: Unknown optimizer')
            sys.exit(1)

        optimizer = optimizer.minimize(self.loss, global_step=global_step)
        return optimizer, global_step

    def __init__(
        self,
        input_shape=(None, None),

        # network structure args
        recurrent_activation='tanh',
        dense_activation='relu',
        recurrent_initializer='glorot_uniform',
        dense_kernel_initializer='glorot_uniform',
        dense_bias_initializer='zeros',
        dropout=0.1,

        # loss
        loss_type='psd',
        lowcut=3.0,
        highcut=60.0,
        fs=512,

        # optimizer args
        optimizer=None,
        lr=None,
        momentum=0.0,
        nesterov=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        rho=None,
        decay=None,

        **kargs):

        T, D = input_shape
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, D, T), name='input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='labels')
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        self.layers = []
        self.params = {}

        # prediction
        reshape = tf.split(self.input, T, 2)
        reshape = [tf.reshape(i, (tf.shape(self.input)[0], D)) for i in reshape]
        self.layers.append(reshape)
        self.predictions = self._build_graph(
            reshape,
            recurrent_activation=recurrent_activation,
            dense_activation=dense_activation,
            recurrent_initializer=recurrent_initializer,
            dense_kernel_initializer=dense_kernel_initializer,
            dense_bias_initializer=dense_bias_initializer,
            dropout=dropout)

        # loss function
        self.params['loss'] = {
        'nfft': tf.placeholder(dtype=tf.int32, name='nfft'),
        'loss_type': loss_type,
        'lowcut': lowcut,
        'highcut': highcut,
        'fs': fs,
        'psd_scale':1
        }
        self.loss = self._loss(**self.params['loss'])
        tf.summary.scalar('loss', self.loss)

        # optimizer
        self.params['optimizer'] = {
        'optimizer': optimizer,
        'lr': lr,
        'nesterov': nesterov,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'epsilon': epsilon,
        'rho': rho,
        'decay':decay,
        }

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer, _ = self._optimizer(**self.params['optimizer'])
