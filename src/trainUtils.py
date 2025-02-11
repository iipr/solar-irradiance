import numpy as np
import pandas as pd
import tables as tb
import time, os
from keras.utils import Sequence
from keras.callbacks import Callback


class HistoryCallback(Callback):
    '''
    Optional functions to define can be:
       - on_(epoch|batch|train)_(begin|end)
    Expected arguments:
       - (self, (epoch|batch), logs={})
    '''
    def __init__(self, mod_name, log_file, month=None):
        self.mod_name = mod_name
        self.log_file = log_file
        self.month = month

    def on_train_begin(self, logs={}):
        self.epochs = []
        self.log_list = []
        self.times = []
        self.months = []
        # Get current time in UTC+1
        now = time.strftime('%A, %d %b %Y %H:%M:%S', time.gmtime(time.time() + 3600 * 1))
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\n\nStarting to train model {} on {}...'.format(self.mod_name, now))

    def on_epoch_begin(self, epoch, logs={}):
        self.init_time = time.time()
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nStarting epoch {}, month {}...'.format(epoch, self.month))

    def on_epoch_end(self, epoch, logs={}):
        end_time = round(time.time() - self.init_time)
        self.epochs.append(epoch)
        self.log_list.append(logs.copy())
        self.times.append(end_time)
        self.months.append(self.month)
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nIt took {}s'.format(end_time))

    def on_train_end(self, logs={}):
        hist = pd.DataFrame()
        hist['epoch'] = self.epochs
        if self.month is not None:
            hist['month'] = self.months
        hist['duration [s]'] = self.times
        # Iterate on log keys (typically: loss, val_loss...)
        for col in self.log_list[0].keys():
            hist[col] = [log[col] for log in self.log_list]
        history_file = self.log_file.replace('logs.txt', 'hist.csv')
        if os.path.exists(history_file):
            prev_hist = pd.read_csv(history_file)
            hist = pd.concat([prev_hist, hist])
        hist.set_index('epoch').to_csv(history_file, index=True)

    
class DataGenerator(Sequence):
    '''
    Data generator for Keras (fit_generator). Based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    
    def __init__(self, days_list, timestep, forecast_horizon,
                 dataset_path, group, batch_size, X_reshape,
                 stateful=False, shuffle=False):
        '''Initialization of the generator object'''
        # For time and horizons
        self.days_list = days_list
        if shuffle: # note that days_list should be a np.array
            np.random.shuffle(self.days_list)
        self.n_days = len(self.days_list)
        # From 5am to 8pm every 1/10/30/60 seconds
        time_gran = os.path.basename(dataset_path).split('_')[-1][:3]
        time_gran = int(time_gran[:2]) * (60 if time_gran[-1] == 'm' else 1)
        self.day_length = 54001 // time_gran
        self.day_pointer = 0
        self.timestep = timestep
        self.ts_pointer = 0
        self.forecast_horizon = forecast_horizon
        # It is assumed that all days have the same length
        self.n_preds_day = self.day_length - self.timestep - self.forecast_horizon[-1] + 1
        # For training and prediction
        self.batch_size = batch_size
        self.curr_bs = self.batch_size
        self.n_to_read = self.batch_size + self.timestep - 1
        self.X_reshape = (self.batch_size, *X_reshape[1:])
        self.y_reshape = (self.batch_size, len(self.forecast_horizon), *X_reshape[2:])
        self.shuffle = shuffle
        self.stateful = stateful
        self._round = np.floor if self.stateful else np.ceil
        # For files and paths
        self.dataset_path = dataset_path
        self.group = group + days_list[0]
        # Starting point to read data depends on being 2d or 3d
        self.start = 0 if 'map' in self.dataset_path else 1

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        batches_per_day = int(self._round((self.n_preds_day) / self.batch_size))
        return self.n_days * batches_per_day

    def __getitem__(self, batch_n):
        '''Generate one batch of data'''
        if self.ts_pointer + self.n_to_read + self.forecast_horizon[-1] > self.day_length:
            self.n_to_read = self.day_length - self.ts_pointer - self.forecast_horizon[-1]
            self.curr_bs = self.n_preds_day - self.ts_pointer
            # If a day is over (or stateful), start off a new one
            if self.n_to_read < self.timestep or self.stateful:
                self.curr_bs = self.batch_size
                self.n_to_read = self.batch_size + self.timestep - 1
                self.ts_pointer = 0
                self.day_pointer += 1
                self.group = self.group[:-2] + self.days_list[self.day_pointer]
        # Read chunk
        X, y = self.__data_generation(np.arange(self.ts_pointer, self.ts_pointer + self.n_to_read))
        # Update the day_pointer for the next batch
        self.ts_pointer += self.batch_size
        return X, y

    def __data_generation(self, idx_x):
        '''Generates data containing the required samples'''
        # It is based on the current batch size, which depends on the batch number
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            X_tot = h5_file.get_node(self.group)[idx_x, self.start:]
            # Reshape, copying slices of lenght = timestep
            X = np.empty((self.curr_bs, *self.X_reshape[1:]))
            for i in range(self.curr_bs):
                X[i] = X_tot[i:i + self.timestep]
            y = np.empty((self.curr_bs, *self.y_reshape[1:]))
            for id_h, horizon in enumerate(self.forecast_horizon):
                idx_y = [idx + horizon for idx in idx_x[self.timestep-1:]]
                y[:, id_h, :] = h5_file.get_node(self.group)[idx_y, self.start:]
        return X, y
