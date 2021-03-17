import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from workalendar.europe import Belgium
from sklearn.preprocessing import MinMaxScaler
import itertools
import argparse
from argparse import RawTextHelpFormatter
# Deep learning:
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import model_from_json

DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
DAYS_OF_WEEK = ['week_1','week_2','week_3','week_4','week_5','week_6','week_7']
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
MINUTES_IN_DAY = MINUTES_IN_HOUR * HOURS_IN_DAY

class DeepModelTS():
    """
    A class to create a deep time series model
    """
    def __init__(
        self,
        data_path: str,
        Y_var: str,
        model_save: str,
        model_load: str,
        import_file_path: str,
        export_file_path: str,
        granularity: str,
        lag: int,
        lag2: int,
        LSTM_layer_depth: int,
        epochs: int,
        batch_size: int,
        train_test_split: int,
        n_test: int,
    ):

        self.data_path = data_path
        self.import_file_path = import_file_path
        self.model_save = model_save
        self.model_load = model_load
        self.granularity = granularity
        self.export_file_path = export_file_path
        self.Y_var = Y_var
        self.lag = lag
        self.lag2 = lag2
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split
        self.n_test = n_test


    @classmethod
    def get_fractional_hour_from_series(self, series: pd.Series) -> pd.Series:
        """
        Return fractional hour in range 0-24, e.g. 12h30m --> 12.5.
        Accurate to 1 minute.
        """
        hour = series.hour
        minute = series.minute
        return hour + minute / MINUTES_IN_HOUR

    @classmethod
    def get_fractional_day_from_series(self, series: pd.Series) -> pd.Series:
        """
        Return fractional day in range 0-1, e.g. 12h30m --> 0.521.
        Accurate to 1 minute
        """
        fractional_hours = self.get_fractional_hour_from_series(series)
        return fractional_hours / HOURS_IN_DAY

    @classmethod
    def get_fractional_year_from_series(self, series: pd.Series) -> pd.Series:
        """
        Return fractional year in range 0-1.
        Accurate to 1 day
        """
        return (series.dayofyear - 1) / DAYS_IN_YEAR

    def preprocess(self, dataframe):
        dataframe.index = pd.to_datetime(dataframe.index)
        # Removing duplicates
        dataframe = dataframe[~dataframe.index.duplicated()]
        #Filling NaN values
        dataframe = dataframe.interpolate()
        # Setting the calendar holidats
        cal = Belgium()
        years = list(range(2014, 2025))
        holidays = []
        for year in years:
            holidays.extend(cal.holidays(year))
        dataframe = dataframe.sort_index()
        # Rename the target column to 'Valeur' for convenience
        dataframe.rename(columns={dataframe.columns[0]: self.Y_var}, inplace=True)

        # Logarithmic transform add 1 for non-negative data (zeros in the series)
        #dataframe[self.Y_var] = log(dataframe[self.Y_var] + 1)

        #working day {0,1}
        dataframe['working day'] = dataframe.index.map(cal.is_working_day).astype(np.float32)
        #fractional hour [0,1]
        dataframe['fractional hour'] = self.get_fractional_day_from_series(dataframe.index)
        # day of week one-hot encoded
        dataframe['day of week'] = dataframe.index.dayofweek + 1
        dataframe['day of week'] = pd.Categorical(dataframe['day of week'], categories=[1,2,3,4,5,6,7], ordered=True)
        dataframe = pd.get_dummies(dataframe,prefix=['week'], columns = ['day of week'], drop_first=False)
        #dataframe = pd.concat([dataframe, pd.DataFrame(columns=DAYS_OF_WEEK)]).fillna(0)
        # fractional day of year
        dataframe['day of year'] = self.get_fractional_year_from_series(dataframe.index)
        # we encode cynical data into two dimensions using a sine and cosine transformations
        def encode(data, col, max_val):
            data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
            return data
        dataframe = encode(dataframe, 'fractional hour', HOURS_IN_DAY)
        dataframe = encode(dataframe, 'day of year', DAYS_IN_YEAR)
        # dropping original columns
        dataframe = dataframe.drop(['fractional hour','day of year'], axis=1)
        return dataframe

    @staticmethod
    def plot_train_history(model):
        '''
        Convergence plots to have an idea on how the training performs
        '''
        loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        plt.figure()
        plt.plot(range(len(loss)), loss, 'b', label='Training loss')
        plt.plot(range(len(val_loss)), val_loss, 'r', label='Validation loss')
        #plt.yscale("log")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.title('Training and validation losses')
        plt.legend()
        plt.show()

    @staticmethod
    def validation(forecasted, real, parameter):
        '''
        compute some important parameters to compare forecasting results
        '''
        value = 0
        value_1 = 0
        value_2 = 0

        if parameter == 'SMAPE':
            for i in range(len(forecasted)):
                if real[i] + forecasted[i] == 0:
                    value += 0
                else:
                    value += ((abs(real[i] - forecasted[i])) / (real[i] + forecasted[i])) * 100
            final_value = value / len(forecasted)

        elif parameter == 'MAPE':
            for i in range(len(forecasted)):
                if real[i] == 0:
                    value += 0
                else:
                    value += (abs(real[i] - forecasted[i]))/real[i]
            final_value = value / len(forecasted) * 100

        elif parameter == 'RMSE':
            for i in range(len(forecasted)):
                value += (real[i] - forecasted[i]) ** 2
            final_value = (value / len(forecasted)) ** (1 / 2)

        elif parameter == 'R':
            for i in range(len(forecasted)):
                value += (real[i] - np.mean(real)) * (forecasted[i] - np.mean(forecasted))
                value_1 += (real[i] - np.mean(real)) ** 2
                value_2 += (forecasted[i] - np.mean(forecasted)) ** 2

            if value_1 == 0 or value_2 == 0:
                final_value = 100
            else:
                final_value = (value / ((value_1 ** (1 / 2)) * (value_2 ** (1 / 2))))*100

        return final_value

    def normalize(self, tensor):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        tensor = self.scaler.fit_transform(tensor)
        return tensor

    def denormalize(self, tensor):
        tensor = self.scaler.inverse_transform(tensor)
        return tensor

    # convert series to supervised learning
    def series_to_supervised(self, data, dropnan=True):
        n_vars = 1
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in [self.lag, self.lag2]:
            cols.append(df[self.Y_var].shift(i))
            names += [(self.Y_var+'(t-%d)' % (i))]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def create_data_for_NN(self):
        """
        A method to create data for the neural network model
        """
        reframed = self.data.merge(self.series_to_supervised(self.data), how='right', left_index=True, right_index=True)
        #print(reframed.head())
        # Normalize with a MinMax Scaler
        reframed = self.normalize(reframed)
        #reframed = np.array(reframed)
        #print(reframed[:10])

        #Assign validation data to fix referencing
        X_val, Y_val = [], []

        test = reframed[-self.n_test:]
        index = len(reframed) - self.n_test
        train = reframed[:round(index * (1-self.train_test_split))]
        if self.train_test_split > 0:
            val = reframed[round(index * (1-self.train_test_split)):index]

        X_train, Y_train = train[:, 1:], train[:, 0]
        if self.train_test_split > 0:
            X_val, Y_val = val[:, 1:], val[:, 0]
        X_test, Y_test = test[:, 1:], test[:, 0]

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        if self.train_test_split > 0:
            X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def save_model(self, model):
        model_json = model.to_json()
        with open(self.model_load+'.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.model_load+'.h5')
        print("Model is saved to disk")

    def load_model(self):
        # load json and create model
        json_file = open(self.model_load+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.model_load+".h5")
        print("Loaded model from disk")

    def LSTModel(self):
        """
        A method to fit the LSTM model
        """
        # Getting the data
        X_train, X_val, X_test, Y_train, Y_val, Y_test = self.create_data_for_NN()
        # Defining the model
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(self.LSTM_layer_depth, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='msle')
        # Setting up early stopping
        earlyStop=EarlyStopping(monitor="val_loss",verbose=1,mode='min',patience=10)
        # Saving training history
        csv_logger = CSVLogger('training_B2_25ep.log', separator=',', append=False)
        # Defining the model parameter dict
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False,
            'callbacks': [earlyStop, csv_logger]
            #'callbacks': [csv_logger]
        }
        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_val, Y_val)
            })
        # Fitting the model
        model.fit(
            **keras_dict
        )
        # Saving the model to the class
        self.model = model
        # Plotting train history
        if self.train_test_split > 0:
            self.plot_train_history(model)
        # Saving the model in json and h5
        self.save_model(self.model)
        return model

    def predict(self) -> list:
        """
        A method to predict using the test data used in creating the class
        """
        yhat = []
        # Getting the last n time series
        _, _, X_test, _, _, Y_test = self.create_data_for_NN()
        # Making the prediction list
        yhat = self.model.predict(X_test)
        # Reshape for merging with predictions
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, X_test), axis=1)
        # denormalize
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        # invert scaling for actual
        Y_test = Y_test.reshape((len(Y_test), 1))
        inv_y = np.concatenate((Y_test, X_test), axis=1)
        # denormalize
        inv_y = self.scaler.inverse_transform(inv_y)
        # extract the actual and predicted values
        act = [i[0] for i in inv_y] # last element is the predicted average energy
        pred = [i[0] for i in inv_yhat] # last element is the actual average energy
        # Reverse the log transformation and substract by one if the data contains zero
        #act = np.exp(act) - 1
        #pred = np.exp(pred) - 1
        return act, pred

    def results(self):
        expected, predictions = self.predict()# Constructing the forecast dataframe
        print('RMSE: %f [kWh]' % self.validation(predictions,expected, 'RMSE'))
        print('MAPE: %f %%' % self.validation(predictions,expected, 'MAPE'))
        print('\n')

    def plot_test(self):
        expected, predictions = self.predict()# Constructing the forecast dataframe
        fc = self.data.tail(len(expected)).copy()
        print('RMSE: %f [kWh]' % self.validation(predictions,expected, 'RMSE'))
        print('MAPE: %f %%' % self.validation(predictions,expected, 'MAPE'))
        #print(expected)
        #print(predictions)
        # Ploting the forecasts
        plt.figure(figsize=(12, 8))
        #for dtype in ['Valeur', 'forecast']:
        plt.plot(fc.index, expected, label='Valeur',alpha=0.7)
        plt.plot(fc.index, predictions, label='forecast',alpha=0.7)
        plt.legend()
        plt.grid()
        plt.gca().set(ylabel='Consumption [kWh]', xlabel='timestamp')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Consumption for test data", fontsize=20)
        plt.show()

    def predict_n_ahead(self, data_input, n_ahead: int):
        # Set up a dataset with n timestamps ahead
        dates = pd.date_range(data_input.index[-1], periods = n_ahead+1, freq=self.granularity)[1:]
        test = data_input.append(pd.DataFrame(index=dates))
        test.index=data_input.index.union(dates)
        # Add all temporal features
        test = self.preprocess(test)
        # Merge temporal features with lags
        test = test.merge(self.series_to_supervised(test), how='right', left_index=True, right_index=True)
        # Normalize with a MinMax scaler
        test = self.normalize(test)
        # Take out the forecasted metric
        test = test[:, 1:]
        # Reshape the tensor to LSTM input
        test = test.reshape((test.shape[0], 1, test.shape[1]))
        # Forecast for n_ahead steps ahead
        yhat = []
        yhat = self.model.predict(test)
        # Reshape for merging with predictions
        test = test.reshape(test.shape[0], test.shape[2])
        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, test), axis=1)
        # denormalize
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        # last element is the predicted average energy
        yhat = [i[0] for i in inv_yhat]
        return yhat[-n_ahead:]

    def evaluate_n_ahead(self, n_ahead: int):
        data_user = pd.read_csv(self.import_file_path, index_col=0)
        data_user.index = pd.to_datetime(data_user.index)
        data_user = data_user.sort_index()
        data_temp = data_user
        # take the user input dataset as the one we predict ahead for
        yhat = []
        predictions = []
        # Rename the first column so it is consistent along the code
        data_temp.rename(columns={data_temp.columns[0]: self.Y_var}, inplace=True)
        data_temp.index = pd.to_datetime(data_temp.index)
        # Load model if no preloaded
        if hasattr('self', 'model') == False:
            self.load_model()
        # Slide through the dataset by window of 96 steps and refeed the predictions into inputs
        for i in tf.range(n_ahead//self.lag+1):
            y_hat = self.predict_n_ahead(data_temp, self.lag)
            data_temp = data_temp.append(pd.DataFrame(y_hat, columns=['Valeur'], index=pd.date_range(data_temp.index[-1], periods = self.lag+1, freq=self.granularity)[1:]))
            data_temp.drop(data_temp.head(self.lag).index, inplace=True)
            predictions.extend(y_hat)
        # Create a dataset for predictions
        predictions = predictions[:n_ahead]
        dates = pd.date_range(data_user.index[-1], periods = n_ahead+1, freq=self.granularity)[1:]
        test = pd.DataFrame(predictions)
        test.index = dates
        test.index = pd.to_datetime(test.index)
        # Save to a csv
        test.to_csv(self.export_file_path, index=True)
        plt.figure(figsize=(25, 10))
        plt.grid()
        plt.gca().set(ylabel='Consumption [kWh]', xlabel='timestamp')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Consumption forecast for given days ahead", fontsize=20)
        #plt.plot(test_range.index, test_range.loc[:,"Valeur"], color='orange', label='test', alpha=0.7)
        plt.plot(data_user.index, data_user.loc[:,"Valeur"], color='b', label='user input data', alpha=0.5)
        plt.plot(dates, predictions, color='black', linestyle='--', linewidth=3, label='Forecaster model',alpha=0.7)
        plt.legend(prop={'size': 20})
        plt.show()
        #return test

    def configuration(self):
        epilogue_usage = """
        Use cases examples:
        Import the data './building1_input.csv' and use the preloaded model "model_B!_complete" to predict 672 steps ahead and save to './predictions.csv'. The model is loaded from '/../data/model/':
        python forecaster.py -F -i /example_mordor/environment/sauron_eye_consumer_24h.csv -n 196 -e ./predictions.csv -M model_mordor\n

        Train the new model "model_B1_new" on the imported data './Consumption_15min.csv' with n steps for test. The model is saved to '/../data/model/':
        python forecaster.py -T -i /example_mordor/environment/sauron_eye_consumer_24h.csv -M model_mordor -t 1008\n

        """
        parser = argparse.ArgumentParser(description='Make energy consumption forecasts and . Read the example to understand how it works', epilog= epilogue_usage,formatter_class=RawTextHelpFormatter)

        parser.add_argument('-F', '--forecast', action='store_true',  help='Download data from copernicus database', default=False)
        parser.add_argument('-T', '--train',  action='store_true', help='Format data from netcdf file to environment file', default=False)
        parser.add_argument('-i', '--imp_dir', required=True, help='Import a timeseries')
        parser.add_argument('-e', '--exp_dir', help='Export a forecasted timeseries')
        parser.add_argument('-n', '--steps_ahead', help='Number of time steps to forecast')
        parser.add_argument('-t', '--steps_test', help='Number of last time steps in the trainig set'
                                                        'to keep for a test dataset')
        parser.add_argument('-M', '--model', help='Name of model to save/load')


        args = parser.parse_args()
        #print(args)

        # FORECAST
        if args.forecast:
            print('The imported csv is: ',args.imp_dir)
            print('Number of timesteps ahead to forecast is: ',args.steps_ahead)
            print('\n')
            if args.model:
                print('You decided to use a trained model:',args.model)
                print('\n')
                self.model_load = DATA_DIR+'model/'+args.model
            self.import_file_path = DATA_DIR+args.imp_dir
            self.export_file_path = args.exp_dir
            self.evaluate_n_ahead(int(args.steps_ahead))
            print('\n')
            print('The forecasted timeseries is exported to',args.exp_dir)
            #return test

        # TRAIN
        if args.train:
            print('The imported csv is: ',args.imp_dir)
            print('\n')
            print('You decided to train a new model',args.model)
            print('\n')
            self.model_load = DATA_DIR+'model/'+args.model
            self.data_path = DATA_DIR[:-1]+args.imp_dir
            self.n_test = int(args.steps_test)
            self.data = pd.read_csv(self.data_path, index_col=0)
            self.data = self.preprocess(self.data)
            self.data = self.data.asfreq(self.data.index.freq or to_offset(self.data.index[1] - self.data.index[0]).freqstr)
            self.granularity = self.data.index.inferred_freq
            print('Data granularity is ',self.granularity)
            print('\n')
            if self.granularity == '15T':
                self.lag = 96
                self.lag2 = 672
            if self.granularity == 'H':
                self.lag = 24
                self.lag2 = 168
            self.model = deep_learner.LSTModel()
            print('\n')
            print('Number of timesteps to use in a test dataset: ',args.steps_test)
            self.plot_test()


        print('\n')
        print('The end')
        print('\n')

if __name__ == "__main__":
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"
    deep_learner = DeepModelTS(
    # Here I initialize some settings, these are default ones if no user input
    # USER INPUT SETTINGS
    Y_var = 'Valeur',
    model_load = DATA_DIR+"model_B1_complete",
    granularity = '15T',
    import_file_path = './building1_input.csv',
    export_file_path = './predictions.csv',
    # ADVANCED TRAINING SETTINGS
    data_path = DATA_DIR+'Consumption_15min.csv',
    model_save = DATA_DIR+"model_B1_complete",
    lag = 96,
    lag2 = 672,
    LSTM_layer_depth = 50,
    epochs = 100,
    batch_size = 128,
    train_test_split = 0.15,
    n_test = 672*4
    )
    deep_learner.configuration()
