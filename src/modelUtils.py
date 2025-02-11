from keras import layers, models, optimizers, utils


class Models():
    '''
    Class that contains the hardcoded models.
    Also allows to load a pre-trained model from an HDF5 file.
    '''
    
    def __init__(self, timestep, n_horizons, n_sensors,
                 sensor_shape=(10, 10),
                 loss='mse', metrics=['mae', 'mse'],
                 optimizer=optimizers.Adam, lr=1e-3):
                 #optimizer=optimizers.Adadelta, lr=1.0):
        self.timestep = timestep
        self.n_horizons = n_horizons
        self.n_sensors = n_sensors
        self.sensor_shape = sensor_shape
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr = lr
        self.models_d = {'fc0': self.__fc0,             'fc2': self.__fc2,
                         'rnn0': self.__rnn0,           'rnn1': self.__rnn1,           'rnn2': self.__rnn2, 
                         'lstm0': self.__lstm0,         'lstm1': self.__lstm1,
                         'biLstm0': self.__biLstm0,     'biLstm1': self.__biLstm1,
                         'convLstm0':self. __convLstm0, 'convLstm1': self.__convLstm1,
                         'conv0': self.__conv0,         'conv1': self.__conv1,         'conv3d0': self.__conv3d0,
                         'lstmSt0': self.__lstmSt0}

    def get_models_list(self):
        return list(self.models_d.keys())
    
    def model(self, mod_name):
        if mod_name in self.models_d.keys():
            return self.models_d[mod_name]()
        print('Model not found!')
        return None

    def load_model(self, model_path, do_compile=False):
        model = models.load_model(model_path)
        if do_compile:
                self.__compile_model(model)
        return model

    def plot_model(self, model, fpath):
        utils.plot_model(model, to_file=fpath,
                         show_shapes=True, show_layer_names=False)
        print('\n\tModel graph saved to {}'.format(fpath))

    def __compile_model(self, model):
        model.compile(optimizer=self.optimizer(lr=self.lr),
                      loss=self.loss, metrics=self.metrics)
    
    def __fc0(self):
        model = models.Sequential()
        model.add(layers.Reshape((self.timestep * self.n_sensors,),
                                 input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dense(68, activation='relu'))
        model.add(layers.Dense(136, activation='relu'))
        model.add(layers.Dense(self.n_horizons * self.n_sensors, activation='linear'))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __fc2(self):
        model = models.Sequential()
        model.add(layers.Reshape((self.timestep * self.n_sensors,),
                                 input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dense(68, activation='relu'))
        model.add(layers.Dense(136, activation='relu'))
        model.add(layers.Dense(68, activation='relu'))
        model.add(layers.Dense(self.n_horizons * self.n_sensors, activation='linear'))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __rnn0(self):
        model = models.Sequential()
        model.add(layers.SimpleRNN(16, activation='linear',
                                   input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __rnn1(self):
        model = models.Sequential()
        model.add(layers.SimpleRNN(16, activation='linear',
                                   input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dense(34))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __rnn2(self):
        model = models.Sequential()
        model.add(layers.SimpleRNN(16, activation='linear',
                                   input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(34))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm0(self):
        model = models.Sequential()
        model.add(layers.LSTM(16, input_shape=(self.timestep, self.n_sensors),
                              return_sequences=True))
        model.add(layers.LSTM(34))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm1(self):
        model = models.Sequential()
        model.add(layers.LSTM(16, input_shape=(self.timestep, self.n_sensors),
                              return_sequences=True, dropout=0.3))
        model.add(layers.LSTM(34, dropout=0.3))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __biLstm0(self):
        model = models.Sequential()
        model.add(layers.Bidirectional(layers.LSTM(64), 
                                       input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __biLstm1(self):
        model = models.Sequential()
        model.add(layers.Bidirectional(layers.LSTM(64), 
                                       input_shape=(self.timestep, self.n_sensors)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='tanh'))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model
    
    def __convLstm0(self):
        model = models.Sequential()
        # Reshape to add dimension: n_channels
        model.add(layers.Reshape((self.timestep, *self.sensor_shape, 1),
                                 input_shape=(self.timestep, *self.sensor_shape)))
        model.add(layers.convolutional_recurrent.ConvLSTM2D(filters=4, kernel_size=(2, 2),
                                                            return_sequences=True))
        model.add(layers.Flatten())
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.sensor_shape[0] * self.sensor_shape[1]))
        model.add(layers.Reshape((self.n_horizons, *self.sensor_shape)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __convLstm1(self):
        model = models.Sequential()
        # Reshape to add dimension: n_channels
        model.add(layers.Reshape((self.timestep, *self.sensor_shape, 1),
                                 input_shape=(self.timestep, *self.sensor_shape)))
        model.add(layers.convolutional_recurrent.ConvLSTM2D(filters=4, kernel_size=(2, 2),
                                                            return_sequences=True))
        model.add(layers.convolutional_recurrent.ConvLSTM2D(filters=4, kernel_size=(2, 2),
                                                            return_sequences=True))
        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.sensor_shape[0] * self.sensor_shape[1]))
        model.add(layers.Reshape((self.n_horizons, *self.sensor_shape)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __conv0(self):
        model = models.Sequential()
        # Permute dimenstions:
        # (None, timestep, sensor_shape) -> (None, sensor_shape, timestep)
        model.add(layers.Permute((2, 3, 1), input_shape=(self.timestep, *self.sensor_shape)))
        model.add(layers.Conv2D(filters=4, kernel_size=(2, 2)))
        model.add(layers.Conv2D(filters=4, kernel_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.sensor_shape[0] * self.sensor_shape[1]))
        model.add(layers.Reshape((self.n_horizons, *self.sensor_shape)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __conv1(self):
        model = models.Sequential()
        # Permute dimenstions:
        # (None, timestep, sensor_shape) -> (None, sensor_shape, timestep)
        model.add(layers.Permute((2, 3, 1), input_shape=(self.timestep, *self.sensor_shape)))
        model.add(layers.Conv2D(filters=10, kernel_size=(2, 2)))
        model.add(layers.Conv2D(filters=10, kernel_size=(1, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.sensor_shape[0] * self.sensor_shape[1]))
        model.add(layers.Reshape((self.n_horizons, *self.sensor_shape)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __conv3d0(self):
        model = models.Sequential()
        # Reshape to add dimension: n_channels
        model.add(layers.Reshape((self.timestep, *self.sensor_shape, 1),
                                 input_shape=(self.timestep, *self.sensor_shape)))
        # Permute dimenstions:
        # (None, timestep, sensor_shape, 1) -> (None, sensor_shape, timestep, 1)
        model.add(layers.Permute((2, 3, 1, 4)))
        model.add(layers.Conv3D(filters=4, kernel_size=(2, 2, 2),
                                input_shape=(self.timestep, *self.sensor_shape, 1)))
        model.add(layers.Conv3D(filters=4, kernel_size=(2, 2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.sensor_shape[0] * self.sensor_shape[1]))
        model.add(layers.Reshape((self.n_horizons, *self.sensor_shape)))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstmSt0(self):
        model = models.Sequential()
        model.add(layers.LSTM(16, input_shape=(self.timestep, self.n_sensors),
                              batch_size=2**5, stateful=True,
                              return_sequences=True))
        model.add(layers.LSTM(34))
        model.add(layers.Dense(32))
        model.add(layers.Dense(self.n_horizons * self.n_sensors))
        model.add(layers.Reshape((self.n_horizons, self.n_sensors)))
        # Compile model and return
        self.__compile_model(model)
        return model

    # Timestep
    def get_timestep(self): 
        return self.timestep 
    def set_timestep(self, x): 
        self.timestep = int(x)

    # Number of forecast horizons
    def get_n_horizons(self): 
        return self.n_horizons
    def set_n_horizons(self, x): 
        self.n_horizons = int(x)

    # Number of sensors
    def get_n_sensors(self): 
        return self.n_sensors
    def set_n_sensors(self, x): 
        self.n_sensors = int(x)

    # Sensor shape
    def get_sensor_shape(self):
        return self.sensor_shape
    def set_sensor_shape(self, x):
        self.sensor_shape = x

    # Optimizer
    def get_optimizer(self):
        return self.optimizer
    def set_optimizer(self, x):
        self.optimizer = eval('optimizers.' + x)

    # Loss
    def get_loss(self):
        return self.loss
    def set_loss(self, x):
        self.loss = x

    # Metrics
    def get_metrics(self):
        return self.metrics
    def set_metrics(self, x):
        self.metrics = eval(x)

    # Learning rate
    def get_lr(self): 
        return self.lr 
    def set_lr(self, x): 
        self.lr = float(x)
