import os, time, re
import numpy as np
import pandas as pd
import tables as tb
import deep_playground, modelUtils, plotUtils
from trainUtils import HistoryCallback, DataGenerator
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

import importlib
importlib.reload(plotUtils)
class DeepLearner():
    '''
    Class that encapsulates common tasks when training models.
    '''

    def __init__(self, work_path='..', time_gran='01m', dset='stand',
                 forecast_horizon=[1, 11, 31, 61], # in minutes
                 batch_size=2**7, batch_size_stateful=2**5, timestep=10, # for time awareness
                 shuffle_data_gen=False, train_split=0.8, 
                 test_months=['/2010/04/', '/2010/12/', '/2011/07/']): 
        # For files and paths
        self.work_path = work_path
        self.time_gran = time_gran
        self.models_path = os.path.join(self.work_path, 'models', self.time_gran)
        self.data_path = os.path.join(self.work_path, 'data', self.time_gran)
        self.dataset = '{}_{}.h5'.format(dset, self.time_gran)
        self.dataset_path = os.path.join(self.data_path, self.dataset)
        # For training and prediction
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.batch_size_stateful = batch_size_stateful
        self.timestep = timestep
        self.shuffle_data_gen = shuffle_data_gen
        self.train_split = train_split
        self.val_split = 1 - self.train_split
        self.test_months = test_months
        # Target variable and sensors
        self.target_d = {'filled': 'GHI [W/m^2]',
                         'stand': 'Standardised GHI [-]',
                         'csi_haurwitz': 'Clear-sky index [-]',
                         'stand_map08': 'Standardised GHI map 08x08 [-]',
                         'stand_map10': 'Standardised GHI map 10x10 [-]'
                        }
        self.target = self.target_d[dset]
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            self.sensors = eval(h5_file.get_node(self.test_months[0]+'15')._v_attrs['columns'])[1:]
        # Additional stuff
        self.menu = deep_playground.Menu()
        self.helper = deep_playground.Helper()
        self.models = modelUtils.Models(timestep=self.timestep, 
                                        n_horizons=len(self.forecast_horizon),
                                        n_sensors=len(self.sensors))
        self.update_sensor_shape()
        self.plotter = plotUtils.Plotter()
        self.mod_queue = []
        register_matplotlib_converters() # <- to avoid a warning


    #################################################
    #                  TRAINING                     #
    #################################################

    def prepare_training(self):
        title = 'Training phase:'
        opts = {'Create new model.': self.__create_model,
                'Load existing model.': self.__load_model,
                'Check models that will be trained.': self.__models_to_train,
                'Ready for training.': self.__train_models,
                'Back to main menu.': self.menu.exit
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt]()

    def __create_model(self):
        title = 'Available models are:'
        mod_list = self.models.get_models_list()
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                break
        model = self.models.model(opt)
        model.name = self.helper.read('Please enter a new model name')
        self.__prep_model(model)

    def __load_model(self, prep_train=True):
        title = 'Available models are:'
        mod_list = sorted([m for m in os.listdir(self.models_path) 
                           if m[0] != '.' and os.path.isdir(os.path.join(self.models_path, m))])
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                mod_path = os.path.join(self.models_path, opt, opt + '.h5')
                if os.path.exists(mod_path):
                    break
                else:
                    self.helper._print('Wrong file name! -> {}'.format(mod_path))
        do_compile = self.helper.read('Compile model? [y/n]')
        if self.helper.read('See model summary? [y/n]') == 'y':
            model.summary()
        model = self.models.load_model(mod_path, True if do_compile == 'y' else False)
        model.name = opt
        if prep_train: # for training
            self.__prep_model(model)
        else:          # for testing
            return model

    def __prep_model(self, model):
        epochs = self.helper.read('Number of epochs', cast=int)
        self.mod_queue.append((model, epochs))
        self.helper._print('Model added to queue.')
        self.helper._continue()

    def __models_to_train(self):
        for idx, (model, epochs) in enumerate(self.mod_queue):
            self.helper._print('\t{}. {}: {} epochs'.format(idx+1, model.name, epochs))
        self.helper._continue()

    def __train_models(self, do_test=True):
        idx = 0
        while len(self.mod_queue) > 0:
            idx +=1
            model, epochs = self.mod_queue.pop(0)
            self.helper._print('{}. Starting to train model {}...'.format(idx, model.name))
            if not os.path.exists(os.path.join(self.models_path, model.name)):
                os.mkdir(os.path.join(self.models_path, model.name))
            model = self.__train(model, epochs)
            fname = os.path.join(self.models_path, model.name, model.name + '.h5')
            # If model already existed, add those previous epochs
            if os.path.exists(fname):
                with tb.open_file(fname, 'r') as h5_mod:
                    node = h5_mod.get_node('/')
                    epochs += node._v_attrs['epochs']
            model.save(fname)
            # For reproducibility:
            self.__add_meta(fname, model.name, epochs)
            # After training, do tests
            if do_test:
                self.__plot_model(model)
                self.__plot_loss(model)
                self.__test(model)
                self.__compute_skill(model)
                if 'map' not in self.target:
                    with tb.open_file(self.dataset_path, mode='r') as h5_file:
                        for test_month in self.test_months:
                            _, y, m, _ = test_month.split('/')
                            d = np.random.choice(list(h5_file.root[y][m]._v_children), 1)
                            self.__test_date(model, '{}/{}/{}'.format(*d, m, y))

    def __train(self, model, epochs):
        # Define parameters for the generator
        params = {'timestep': self.timestep,
                  'forecast_horizon': self.forecast_horizon,
                  'dataset_path': self.dataset_path,
                  'batch_size': self.batch_size_stateful if model.stateful else self.batch_size,
                  'X_reshape': model.layers[0].input_shape,
                  'stateful': model.stateful
                 }
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            for epoch in range(epochs):
                for year in h5_file.root._v_children:
                    for month in h5_file.root[year]._v_children:
                        params['group'] = '/{}/{}/'.format(year, month)
                        if params['group'] in self.test_months:
                            continue
                        days = h5_file.root[year][month]._v_children
                        n_days = h5_file.root[year][month]._v_nchildren
                        train_days = list(days)[0:int(n_days * self.train_split)]
                        val_days = list(days)[int(n_days * self.train_split):n_days]
                        train_generator = DataGenerator(train_days, **params)
                        val_generator = DataGenerator(val_days, **params)
                        log_file = os.path.join(self.models_path, model.name, model.name + '_logs.txt')
                        # For the current epoch, year and month, call fit 
                        model.fit_generator(generator=train_generator,
                                            validation_data=val_generator, 
                                            shuffle=self.shuffle_data_gen,
                                            initial_epoch=epoch, epochs=epoch+1,
                                            callbacks=[HistoryCallback(mod_name=model.name,
                                                                       log_file=log_file,
                                                                       month=month)],
                                            use_multiprocessing=True, workers=0)
        return model

    def __add_meta(self, fname, mod_name, epochs):
        # Add metadata to the model
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs['name'] = mod_name
            node._v_attrs['epochs'] = epochs
            # Data
            node._v_attrs['time_gran'] = self.time_gran
            node._v_attrs['forecast_horizon'] = self.forecast_horizon
            node._v_attrs['dataset'] = self.dataset
            # Model
            node._v_attrs['batch_size'] = self.batch_size
            node._v_attrs['batch_size_stateful'] = self.batch_size_stateful
            node._v_attrs['timestep'] = self.timestep
            node._v_attrs['shuffle_data_gen'] = self.shuffle_data_gen
            node._v_attrs['train_split'] = self.train_split
            node._v_attrs['val_split'] = self.val_split
            node._v_attrs['test_months'] = self.test_months
            # Optimizer
            node._v_attrs['optimizer'] = self.models.get_optimizer()
            node._v_attrs['loss'] = self.models.get_loss()
            node._v_attrs['metrics'] = self.models.get_metrics()
            node._v_attrs['lr'] = self.models.get_lr()

    def __add_attr(self, mod_name, attr, value):
        # Add single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs[attr] = value

    def __get_attr(self, mod_name, attr):
        # Get a single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'r') as h5_mod:
            node = h5_mod.get_node('/')
            return node._v_attrs[attr]


    #################################################
    #                  TESTING                      #
    #################################################

    def prepare_test(self):
        model = self.__load_model(prep_train=False)
        title = 'Testing phase for {}:'.format(model.name)
        opts = {'Plot model graph.': self.__plot_model,
                'Check model hyperparameters.': self.__model_hyperparams,
                'Plot model loss and metrics.': self.__plot_loss,
                'Test on (unseen) months.': self.__test,
                'Compute forecast skill.': self.__compute_skill,
                'Robustness test.': self.__robustness_test,
                'Test and plot a certain day.': self.__test_date,
                'Back to main menu.': self.menu.exit,
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt](model)
                self.helper._continue()

    def __plot_model(self, model):
        aux_path = os.path.join(self.models_path, model.name, model.name + '.pdf')
        self.models.plot_model(model, aux_path)

    def __model_hyperparams(self, model):
        fname = os.path.join(self.models_path, model.name, model.name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            for idx, attr in enumerate(node._v_attrs._v_attrnames):
                if '_config' not in attr:
                    self.helper._print('\t{:02}. {}: {}'.format(idx+1, attr, node._v_attrs[attr]))

    def __plot_loss(self, model):
        # Load history file
        aux_path = os.path.join(self.models_path, model.name, model.name)
        hist = pd.read_csv(aux_path + '_hist.csv')
        days = hist['duration [s]'].sum() // (24 * 3600)
        train_duration = '{} days and {}'.format(days, time.strftime('%H:%M:%S', 
                                                                     time.gmtime(hist['duration [s]'].sum())))
        self.helper._print('Total training time for model {} was {}'.format(model.name, train_duration))
        self.__add_attr(model.name, 'train_duration', train_duration)
        loss_names = {'mse': 'Mean Squared Error',
                      'val_mse': 'Mean Squared Error',
                      'mae': 'Mean Absolute Error',
                      'val_mae': 'Mean Absolute Error',
                      'acc': 'Accuracy',
                      'val_acc': 'Accuracy',
                      'categorical_crossentropy': 'Categorical Crossentropy',
                      'msle': 'Mean Squared Logarithmic Error',
                      'val_msle': 'Mean Squared Logarithmic Error'
                     }
        series_d = dict()
        # Here we reverse the columns for a better order in the plot and legend
        for hkey in reversed(hist.columns[3:]): # after epoch, month & duration
            if 'loss' in hkey:
                continue
            metric_name = loss_names[hkey]
            if self.models.get_loss() in hkey:
                metric_name += ' - Loss'
            t_set = 'Validation set' if 'val_' in hkey else 'Training set'
            if metric_name in series_d.keys():
                series_d[metric_name][t_set] = hist.loc[:, hkey].values
            else:
                series_d[metric_name] = {t_set: hist.loc[:, hkey].values}
        self.plotter.plot_series(series_d, ('month', ''), int_ticker=True,
                                 out_path=aux_path + '_loss.pdf')

    def __test(self, model):
        # Define parameters for the generator
        params = {'timestep': self.timestep,
                  'forecast_horizon': self.forecast_horizon,
                  'dataset_path': self.dataset_path,
                  'batch_size': self.batch_size_stateful if model.stateful else self.batch_size,
                  'X_reshape': model.layers[0].input_shape,
                  'stateful': model.stateful
                 }
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            for test_month in self.test_months:
                self.helper._print('Testing {}...'.format(test_month))
                params['group'] = test_month
                _, y, m, _ = test_month.split('/')
                days = h5_file.root[y][m]._v_children
                eval_generator = DataGenerator(list(days), **params)
                ret = model.evaluate_generator(eval_generator, verbose=1,
                                               use_multiprocessing=True, workers=0)
                metrics_d = dict(zip(self.models.get_metrics(),
                                     [round(val, 5) for val in ret]))
                self.__add_attr(model.name, 'test_{}_{}'.format(y, m), metrics_d)
                self.helper._print(metrics_d)

    def __compute_skill(self, model):
        # Define parameters for the generator
        params = {'timestep': self.timestep,
                  'forecast_horizon': self.forecast_horizon,
                  'dataset_path': self.dataset_path,
                  'batch_size': self.batch_size_stateful if model.stateful else self.batch_size,
                  'X_reshape': model.layers[0].input_shape,
                  'stateful': model.stateful
                 }
        n_horizons = len(self.forecast_horizon)
        n_sensors = len(self.sensors)
        sse_mod, sse_pers, n_samp = np.zeros((n_horizons, n_sensors)), np.zeros((n_horizons, n_sensors)), 0
        is_irr_map = True if 'stand_map' in self.dataset else False
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            for test_month in self.test_months:
                params['group'] = test_month
                _, y, m, _ = test_month.split('/')
                days = h5_file.root[y][m]._v_children
                n_days = h5_file.root[y][m]._v_nchildren
                pred_generator = DataGenerator(list(days), **params)
                for n_batch, (X_b, y_b) in enumerate(pred_generator):
                    y_pred = model.predict_on_batch(X_b)
                    if is_irr_map:
                        # When working with irradiance maps, select certain nodes
                        X_b = X_b[:, :, self.xid, self.yid]
                        y_b = y_b[:, :, self.xid, self.yid]
                        y_pred = y_pred[:, :, self.xid, self.yid]
                    X_b = X_b.reshape((*X_b.shape[:2], -1))
                    y_b = y_b.reshape((*y_b.shape[:2], -1))
                    y_pred = y_pred.reshape((*y_pred.shape[:2], -1))
                    sse_mod += ((y_b - y_pred) ** 2).sum(axis=0)
                    for id_h, horizon in enumerate(self.forecast_horizon):
                        sse_pers[id_h] += ((y_b[:, id_h, :] - X_b[:, -1, :]) ** 2).sum(axis=0)
                    n_samp += y_b.shape[0]
        # Compute and save the RMSE for the model and the persistence model
        rmse_mod = np.sqrt(sse_mod / n_samp)
        rmse_pers = np.sqrt(sse_pers / n_samp)
        self.__add_attr(model.name, 'model_rmse', rmse_mod)
        self.__add_attr(model.name, 'persistent_rmse', rmse_pers)
        # Compute and save the skill per sensor and horizon
        skill = (1 - rmse_mod / rmse_pers) * 100
        skill_median = np.median(skill, axis=1)
        skill_mean = np.mean(skill)
        self.__add_attr(model.name, 'forecast_skill', skill)
        self.__add_attr(model.name, 'skill_median', skill_median)
        self.__add_attr(model.name, 'skill_mean', skill_mean)
        self.helper._print('Median of skills per horizon: {}%'.format(skill_median))
        self.helper._print('Mean of all skills: {}%'.format(skill_mean))
        self.plotter.plot_boxplot(skill.T, x_labels=self.forecast_horizon,
                                  labels=('Forecast horizon [min]', 'Forecast skill [%]'),
                                  title='Skill variation among sensors for each horizon',
                                  out_path=os.path.join(self.models_path, model.name, model.name + '_skill.pdf'))

    def __robustness_test(self, model, up_to=16, n_reps=10):
        # Define parameters for the map generator
        params_m = {'timestep': self.timestep,
                    'forecast_horizon': self.forecast_horizon,
                    'dataset_path': self.dataset_path,
                    'batch_size': self.batch_size_stateful if model.stateful else self.batch_size,
                    'X_reshape': model.layers[0].input_shape,
                    'stateful': model.stateful
                   }
        # Define parameters for the original dataset (before interpolation)
        params_s = params_m.copy()
        params_s['dataset_path'] = re.sub('map\d+_', '', self.dataset_path)
        params_s['X_reshape'] = (None, self.timestep, len(self.sensors))
        # Other required stuff
        reps = ['rep_{:02}'.format(idx) for idx in range(n_reps)]
        rob_path = os.path.join(self.models_path, model.name, 'robustness-test')
        if not os.path.exists(rob_path):
            os.mkdir(rob_path)
        n_horizons = len(self.forecast_horizon)
        n_sensors = len(self.sensors)
        rmse_m = self.__get_attr(model.name, 'model_rmse')
        # To keep the results
        df_sk = pd.DataFrame(columns=['n_excl', 'perc_excl'] +\
                                     ['rep_{:02}'.format(idx) for idx in range(n_reps)])
        df_sk['n_excl'] = np.arange(1, up_to+1)
        df_sk['perc_excl'] = (np.arange(1, up_to+1) / n_sensors * 100).round(decimals=2)
        df_sk.set_index('n_excl', inplace=True)
        df_sks, df_exc, df_rmse = df_sk.copy(), df_sk.copy(), df_sk.copy()
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            for n_exc in range(1, up_to+1):
                #self.helper._print('Excluding {} sensors...'.format(n_exc))
                for id_rep, n_rep in enumerate(reps):
                    chosen = np.sort(np.random.choice(range(n_sensors), size=n_sensors - n_exc, replace=False))
                    sse_mod, n_samp = np.zeros((n_horizons, n_sensors)), 0
                    for test_month in self.test_months:
                        params_m['group'], params_s['group'] = test_month, test_month
                        _, y, m, _ = test_month.split('/')
                        days = h5_file.root[y][m]._v_children
                        n_days = h5_file.root[y][m]._v_nchildren
                        map_generator = DataGenerator(list(days), **params_m)
                        pred_generator = DataGenerator(list(days), **params_s)
                        for n_batch, ((_, y_m), (X_b, _)) in enumerate(zip(map_generator, pred_generator)):
                            bs = X_b.shape[0]
                            # Select the chosen sensors only
                            X_b = X_b[:, :, chosen]
                            # Interpolate to obtain the input irradiance map, in 2 steps
                            aux = griddata(self.xy[chosen], X_b.transpose((2, 0, 1)).reshape((len(chosen), -1)),
                                           (self.X, self.Y), method='nearest')
                            X_b = aux.reshape((*self.models.get_sensor_shape(), bs, self.timestep)).transpose((2, 3, 0, 1))
                            y_pred = model.predict_on_batch(X_b)
                            # Extract nearest nodes to sensors for error calculation
                            y_m = y_m[:, :, self.xid, self.yid]
                            y_pred = y_pred[:, :, self.xid, self.yid]
                            # Flatten for error calculation
                            y_m = y_m.reshape((*y_m.shape[:2], -1))
                            y_pred = y_pred.reshape((*y_pred.shape[:2], -1))
                            sse_mod += ((y_m - y_pred) ** 2).sum(axis=0)
                            n_samp += y_m.shape[0]
                    rmse_mod = np.sqrt(sse_mod / n_samp)
                    sks = (1 - rmse_mod / rmse_m) * 100
                    sks_median = np.median(sks, axis=1)
                    df_sks.loc[n_exc, n_rep] = str(sks_median.tolist())
                    df_sk.loc[n_exc, n_rep] = np.mean(sks_median)
                    df_exc.loc[n_exc, n_rep] = str([s for idx, s in enumerate(self.sensors) if idx not in chosen])
                    df_rmse.loc[n_exc, n_rep] = str(np.median(rmse_mod, axis=1).tolist())
        df_sks.to_csv(os.path.join(rob_path, 'skill_per_horizon.csv'))
        df_sk.loc[:, 'mean'] = df_sk.loc[:, reps].mean(axis=1)
        df_sk.loc[:, 'std'] = df_sk.loc[:, reps].std(axis=1)
        df_sk.to_csv(os.path.join(rob_path, 'skill_mean.csv'))
        df_exc.to_csv(os.path.join(rob_path, 'excluded.csv'))
        df_rmse.to_csv(os.path.join(rob_path, 'rmse_per_horizon.csv'))
        self.__plot_robustness(df_sks, os.path.join(rob_path, model.name + '_rob_test.pdf'))

    def __plot_robustness(self, df, plt_path):
        up_to, n_reps = df.shape[0], df.shape[1] - 1
        sks = np.empty((up_to, n_reps, len(self.forecast_horizon)))
        for n_row in range(up_to):
            for n_col in range(n_reps):
                for id_h, sk in enumerate(eval(df.iloc[n_row, n_col+1])):
                    sks[n_row, n_col, id_h] = sk
        series_d = dict(zip(['{}min'.format(h) for h in self.forecast_horizon],
                            (*sks.mean(axis=1).T,)))
        self.plotter.plot_series({'Skill with respect to the original model': series_d},
                                 ('Number of excluded sensors', 'Worsening [%]'), int_ticker=True,
                                 style='o-', out_path=plt_path)

    def __predict(self, model, day, month, year):
        # Define parameters for the generator
        params = {'timestep': self.timestep,
                  'forecast_horizon': self.forecast_horizon,
                  'dataset_path': self.dataset_path,
                  'batch_size': self.batch_size_stateful if model.stateful else self.batch_size,
                  'X_reshape': model.layers[0].input_shape,
                  'stateful': model.stateful,
                  'group': '/{}/{}/'.format(year, month)
                 }
        with tb.open_file(self.dataset_path, mode='r') as h5_file:
            pred_generator = DataGenerator([day], **params)
            pred = model.predict_generator(pred_generator, verbose=1,
                                           use_multiprocessing=True, workers=0)
            hours = np.empty((pred.shape[0], len(self.forecast_horizon)), dtype=object)
            true = np.empty((pred.shape[0], len(self.forecast_horizon), len(self.sensors)))
            for id_h, horizon in enumerate(self.forecast_horizon):
                left = self.timestep + horizon - 1
                slc = slice(left, left + pred.shape[0])
                hours_hor = h5_file.get_node('/{}/{}/{}'.format(year, month, day))[slc, 0]
                hours[:, id_h] = np.array([datetime.utcfromtimestamp(hour) for hour in hours_hor])
                true[:, id_h, :] = h5_file.get_node('/{}/{}/{}'.format(year, month, day))[slc, 1:]
        return hours, pred, true

    def __test_date(self, model, date=None):
        if not date:
            date = self.helper.read('Date (dd/mm/yyyy)')
        d, m, y = date.split('/')
        hours, pred, true = self.__predict(model, d, m, y)
        # Obtain min and max values for the limits of the plot
        min_val = min(min(true.ravel()), min(pred.ravel())) - 0.05
        max_val = max(max(true.ravel()), max(pred.ravel())) + 0.05
        # Create folder for plots
        plt_path = os.path.join(self.models_path, model.name, model.name + '_day_plots')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)
        for id_h, horizon in enumerate(self.forecast_horizon):
            series_d = dict()
            for id_s, sensor in enumerate(self.sensors):
                mse = mean_squared_error(true[:, id_h, id_s], pred[:, id_h, id_s])
                title = 'True vs predicted radiation for {} on {}, \nHorizon = {}min, R^2 = {:0.2}, MSE = {:0.2}'.format(
                    sensor, date, horizon, r2_score(true[:, id_h, id_s], pred[:, id_h, id_s]), mse)
                series_d[title] = {'pred': pred[:, id_h, id_s],
                                   'true': true[:, id_h, id_s]}
            self.plotter.plot_series(series_d, ('Hour', self.target), date_ticker=hours[:, id_h], scale=0.5,
                                     out_path=plt_path + '/{:02}m_{}_series.png'.format(horizon, y+m+d))
            self.plotter.plot_scatters(series_d, ('True ' + self.target, 'Predicted ' + self.target),
                                       scale=0.5, min_max=(min_val, max_val), dims=(int(np.ceil((id_s+1)/2)), 2),
                                       out_path=plt_path + '/{:02}m_{}_scatter.png'.format(horizon, y+m+d))


    #################################################
    #                 UPDATE OPTIONS                #
    #################################################

    def update_options(self):
        title = 'The options that can be changed are:'
        stop = False
        while not stop:
            opts = {'Time granularity: {}'.format(self.get_time_gran()): self.set_time_gran,
                    'Forecast horizon: {}'.format(self.get_forecast_horizon()): self.set_forecast_horizon,
                    'Dataset: {}'.format(self.get_dataset()): self.set_dataset,
                    'Batch size: {}'.format(self.get_batch_size()): self.set_batch_size,
                    'Batch size stateful: {}'.format(self.get_batch_size_stateful()): self.set_batch_size_stateful,
                    'Timestep: {}'.format(self.get_timestep()): self.set_timestep,
                    'Shuffle data generator: {}'.format(self.get_shuffle_data_gen()): self.set_shuffle_data_gen,
                    'Training split: {}'.format(self.get_train_split()): self.set_train_split,
                    'Test months: {}'.format(self.get_test_months()): self.set_test_months,
                    'Optimizer: {}'.format(self.models.get_optimizer()): self.models.set_optimizer,
                    'Loss: {}'.format(self.models.get_loss()): self.models.set_loss,
                    'Metrics: {}'.format(self.models.get_metrics()): self.models.set_metrics,
                    'Learning rate: {}'.format(self.models.get_lr()): self.models.set_lr,
                    'Back to main menu.': self.menu.exit,
                }
            opt = self.menu.run(list(opts.keys()), title=title)
            value = self.helper.read('Value')
            if opt:
                stop = opts[opt](value)


    #################################################
    #                 RESULTS TABLE                 #
    #################################################

    def results_table(self, fname='results.csv'):
        df = pd.DataFrame(columns=['Model', 'Target', 'Time granularity', 'Timesteps',
                                   'Forecast horizon [min]', 'Median of skills [%]',
                                   'Mean of medians [%]', 'Mean of all skills [%]',
                                   'RMSE model', 'RMSE persistence', 'Epochs', 'Loss',
                                   'Optimizer', 'Learning rate', 'Train duration'])
        for idx, model in enumerate(os.listdir(self.models_path)):
            model_path = os.path.join(self.models_path, model, model + '.h5')
            if not os.path.exists(model_path):
                continue
            with tb.open_file(model_path, 'r') as h5_mod:
                node = h5_mod.root
                name = 'unknown'
                for mod_name in self.models.models_d.keys():
                    if mod_name in node._v_attrs['name']:
                        name = mod_name
                        break
                df.loc[idx] = [name,
                               self.target_d[node._v_attrs['dataset'][:-7]],
                               node._v_attrs['time_gran'], node._v_attrs['timestep'],
                               node._v_attrs['forecast_horizon'],
                               np.round(node._v_attrs['skill_median'], 2).tolist(),
                               np.round(np.mean(node._v_attrs['skill_median']), 2),
                               np.round(node._v_attrs['skill_mean'], 2),
                               np.round(np.median(node._v_attrs['model_rmse'], axis=1), 5).tolist(),
                               np.round(np.median(node._v_attrs['persistent_rmse'], axis=1), 5).tolist(),
                               node._v_attrs['epochs'], node._v_attrs['loss'],
                               node._v_attrs['optimizer'].__name__, node._v_attrs['lr'],
                               node._v_attrs['train_duration']
                              ]
        df.sort_values(['Time granularity', 'Target', 'Timesteps', 'Model'], inplace=True)
        df.to_csv(os.path.join(self.models_path, fname), index=False)
        self.helper._print('{} saved at {}'.format(fname, self.models_path))
        self.helper._continue()


    #################################################
    #              GETTERS & SETTERS                #
    #################################################

    # Time granularity
    def get_time_gran(self):
        return self.time_gran
    def set_time_gran(self, x):
        self.time_gran = x
        self.models_path = os.path.join(self.work_path, 'models', self.time_gran)
        self.data_path = os.path.join(self.work_path, 'data', self.time_gran)
        self.dataset = '{}_{}.h5'.format(self.dataset.split('_')[0], self.time_gran)
        self.dataset_path = os.path.join(self.data_path, self.dataset)

    # Forecast horizon
    def get_forecast_horizon(self):
        return self.forecast_horizon
    def set_forecast_horizon(self, x):
        self.forecast_horizon = sorted(eval(x))
        self.models.set_n_horizons(len(self.forecast_horizon))

    # Sensor shape for convolutionals
    def update_sensor_shape(self):
        # Load longitude and latitude for the interpolation
        st_path = os.path.join(self.data_path, '..', 'other', 'stations.txt')
        self.xy = pd.read_csv(st_path, '\t').loc[:, ['Longitude', 'Latitude']].values
        self.longitude = self.xy[:, 0]
        self.latitude = self.xy[:, 1]
        if 'stand_map08' in self.dataset:
            # For 8 by 8:
            self.xid = np.array([3, 4, 5, 3, 6, 7, 0, 2, 2, 2, 2, 1, 1, 1, 0, 1, 2])
            self.yid = np.array([4, 4, 3, 2, 0, 6, 7, 6, 4, 5, 5, 4, 6, 2, 4, 4, 2])
            xnew = np.linspace(min(self.longitude) - 0.0006, max(self.longitude) + 0.0006, 8)
            ynew = np.linspace(min(self.latitude) - 0.0006, max(self.latitude) + 0.0006, 8)
            self.X, self.Y = np.meshgrid(xnew, ynew)
            sensor_shape = (8, 8)
        elif 'stand_map10' in self.dataset:
            # For 10 by 10:
            self.xid = np.array([3, 5, 7, 4, 7, 8, 1, 2, 3, 2, 3, 1, 1, 1, 1, 2, 2])
            self.yid = np.array([6, 6, 4, 3, 1, 7, 8, 7, 5, 6, 6, 5, 7, 3, 5, 5, 3])
            xnew = np.linspace(min(self.longitude) - 0.001, max(self.longitude) + 0.001, 10)
            ynew = np.linspace(min(self.latitude) - 0.001, max(self.latitude) + 0.001, 10)
            self.X, self.Y = np.meshgrid(xnew, ynew)
            sensor_shape = (10, 10)
        else:
            sensor_shape = (len(self.sensors),)
        self.models.set_sensor_shape(sensor_shape)

    # Dataset
    def get_dataset(self):
        return self.dataset
    def set_dataset(self, x):
        self.dataset = '{}_{}.h5'.format(x, self.time_gran)
        self.dataset_path = os.path.join(self.data_path, self.dataset)
        self.target = self.target_d[x]
        self.update_sensor_shape()

    # Batch size
    def get_batch_size(self):
        return self.batch_size
    def set_batch_size(self, x):
        self.batch_size = int(x)

    # Batch size for stateful nets
    def get_batch_size_stateful(self):
        return self.batch_size_stateful
    def set_batch_size_stateful(self, x):
        self.batch_size_stateful = int(x)

    # Timestep
    def get_timestep(self):
        return self.timestep
    def set_timestep(self, x):
        self.timestep = int(x)
        self.models.set_timestep(self.timestep)

    # Shuffle data generator
    def get_shuffle_data_gen(self):
        return self.shuffle_data_gen
    def set_shuffle_data_gen(self, x):
        self.shuffle_data_gen = bool(x)

    # Training split
    def get_train_split(self):
        return self.train_split
    def set_train_split(self, x):
        self.train_split = float(x)
        self.val_split = 1 - self.train_split

    # Test months
    def get_test_months(self):
        return self.test_months
    def set_test_months(self, x):
        self.test_months = eval(x)
