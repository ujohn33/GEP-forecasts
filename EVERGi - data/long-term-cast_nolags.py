import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from workalendar.europe import Belgium
import itertools
import sys
# Deep learning:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import model_from_json
np.set_printoptions(threshold=sys.maxsize)

class DeepModelTS():
    """
    A class to create a deep time series model
    """
    def __init__(
        self,
        Y_var: str,
        model_save: str,
        model_load: str,
        data_path: str,
        import_file_path: str,
        export_file_path: str,
        lag: int,
        lag2: int,
        LSTM_layer_depth: int,
        epochs=10,
        batch_size=256,
        train_test_split=0,
        n_test = 96,
        holi_var = 'working day',
        hour_var_sin = 'hour of day_sin',
        hour_var_cos = 'hour of day_cos',
        day_week_sin = 'day of week_sin',
        day_week_cos = 'day of week_cos',
        day_year_sin = 'day of year_sin',
        day_year_cos = 'day of year_cos',
        minutes_sin = 'minutes_sin',
        minutes_cos = 'minutes_cos',

    ):

        self.data_path = data_path
        self.data =  pd.read_csv(data_path, index_col=0)
        self.import_file_path = import_file_path
        self.data_user = pd.read_csv(import_file_path, index_col=0)
        self.data_user.index = pd.to_datetime(self.data_user.index)
        self.model_save = model_save
        self.model_load = model_load
        self.export_file_path = export_file_path
        self.Y_var = Y_var
        self.holi_var = holi_var
        self.hour_var_sin = hour_var_sin
        self.hour_var_cos = hour_var_cos
        self.day_week_sin = day_week_sin
        self.day_week_cos = day_week_cos
        self.day_year_sin = day_year_sin
        self.day_year_cos = day_year_cos
        self.minutes_sin = minutes_sin
        self.minutes_cos = minutes_cos
        self.lag = lag
        self.lag2 = lag2
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split
        self.n_test = n_test

    @staticmethod
    def preprocess(dataframe):
        dataframe.index = pd.to_datetime(dataframe.index)
        cal = Belgium()
        years = list(range(2014, 2025))
        holidays = []
        for year in years:
            holidays.extend(cal.holidays(year))
        dataframe = dataframe.sort_index()
        dataframe['working day'] = dataframe.index.map(cal.is_working_day)
        dataframe['hour of day'] = dataframe.index.hour
        dataframe['day of week'] = dataframe.index.dayofweek
        dataframe['date'] = dataframe.index.date
        dataframe['day of year'] = dataframe.index.dayofyear
        dataframe['minutes'] = dataframe.index.minute
        # we encode cynical data into two dimensions using a sine and cosine transformations
        def encode(data, col, max_val):
            data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
            return data
        dataframe = encode(dataframe, 'hour of day', 23)
        dataframe = encode(dataframe, 'day of week', 6)
        dataframe = encode(dataframe, 'day of year', 365)
        dataframe = encode(dataframe, 'minutes', 60)
        dataframe = dataframe.drop(['hour of day', 'day of week', 'day of year', 'minutes'], axis=1)
        dataframe = dataframe.fillna(method='ffill')
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

    @staticmethod
    def create_X_Y(ts: list, holiday: list, hour_cos: list, hour_sin: list, week_cos: list, week_sin: list, day_year_cos: list, day_year_sin: list, minute_cos: list, minute_sin: list, lag: int, lag2: int) -> tuple:
        """
        A method to create X and Y matrix from a time series list for the training of
        deep learning models
        """
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                # Substacted 96 for not knowing the day before
                ab = list(itertools.chain([holiday[i + lag]], [hour_cos[i + lag]], [hour_sin[i + lag]], [week_cos[i + lag]], [week_sin[i + lag]], [minute_cos[i + lag]], [minute_sin[i + lag]], [day_year_cos[i + lag]], [day_year_sin[i + lag]]))
                #ab = list(itertools.chain([ts[i+lag - lag]], [ts[i+lag - lag2]], [holiday[i + lag]], [hour_cos[i + lag]], [hour_sin[i + lag]], [week_cos[i + lag]], [week_sin[i + lag]], [month_cos[i + lag]], [month_sin[i + lag]], [minute_cos[i + lag]], [minute_sin[i + lag]]))
                X.append(ab)
                #X.append(ts[i:(i + lag - 96)] + [holiday[i + lag]] + [hour_cos[i + lag]] + [hour_sin[i + lag]] + )

        X, Y = np.array(X), np.array(Y)

        # Reshaping the X array to an LSTM input shape
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        return X, Y

    def create_data_for_NN(
        self,
        use_last_n=None, ahead=False):
        """
        A method to create data for the neural network model
        """
        self.data = self.preprocess(self.data)
        #print(self.data.columns.values)

        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()
        y_holiday = self.data[self.holi_var].tolist()
        y_hour_cos = self.data[self.hour_var_cos].tolist()
        y_hour_sin = self.data[self.hour_var_sin].tolist()
        y_weekday_cos = self.data[self.day_week_cos].tolist()
        y_weekday_sin = self.data[self.day_week_sin].tolist()
        y_day_year_cos = self.data[self.day_year_cos].tolist()
        y_day_year_sin = self.data[self.day_year_sin].tolist()
        y_minute_cos = self.data[self.minutes_cos].tolist()
        y_minute_sin = self.data[self.minutes_sin].tolist()




        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]
            y_holiday = y_holiday[-use_last_n:]
            y_hour_cos = y_hour_cos[-use_last_n:]
            y_hour_sin = y_hour_sin[-use_last_n:]
            y_weekday_cos = y_weekday_cos[-use_last_n:]
            y_weekday_sin = y_weekday_sin[-use_last_n:]
            y_day_year_cos = y_day_year_cos[-use_last_n:]
            y_day_year_sin = y_day_year_sin[-use_last_n:]
            y_minute_cos = y_minute_cos[-use_last_n:]
            y_minute_sin =  y_minute_sin[-use_last_n:]

        # The X matrix will hold the lags of Y
        X, Y = self.create_X_Y(y, y_holiday, y_hour_cos, y_hour_sin, y_weekday_cos, y_weekday_sin, y_day_year_cos, y_day_year_sin, y_minute_cos, y_minute_sin, self.lag, self.lag2)

        # Creating training and test sets
        X_train = X
        X_val = []
        X_test = []

        Y_train = Y
        Y_val = []
        Y_test = []
        if ahead == False:
            if self.train_test_split > 0:
                index = round((len(X) - self.n_test) * self.train_test_split)
                X_train = X[:(len(X) - index)]
                X_val = X[(len(X) - index):- self.n_test]
                X_test = X[-self.n_test:]

                Y_train = Y[:(len(X) - index)]
                Y_val = Y[(len(X) - index):- self.n_test]
                Y_test = Y[-self.n_test:]
        #print(X_train.shape)
        #print(Y_train.shape)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def save_model(self, model):
        model_json = model.to_json()
        with open(self.model_save+'.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.model_save+'.h5')

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
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Setting up early stopping
        earlyStop=EarlyStopping(monitor="val_loss",verbose=1,mode='min',patience=6)

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
        self.plot_train_history(model)
        # Saving the model in json and h5
        self.save_model(self.model)

        return model

    def predict(self) -> list:
        """
        A method to predict using the test data used in creating the class
        """
        yhat = []

        if(self.train_test_split > 0):

            # Getting the last n time series
            _, _, X_test, _, _, _ = self.create_data_for_NN()

            # Making the prediction list
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    def plot_test(self):
        yhat = self.predict()# Constructing the forecast dataframe
        fc = holidata.tail(len(yhat)).copy()
        fc['forecast'] = yhat
        expected = fc.loc[:,'Valeur']
        predictions = fc.loc[:,'forecast']
        print('RMSE: %f [kWh]' % self.validation(predictions,expected, 'RMSE'))
        print('MAPE: %f %%' % self.validation(predictions,expected, 'MAPE'))
        # Ploting the forecasts
        plt.figure(figsize=(12, 8))
        for dtype in ['Valeur', 'forecast']:
            plt.plot(fc.index, fc[dtype],label=dtype,alpha=0.7)
        plt.legend()
        plt.grid()

        plt.gca().set(ylabel='Consumption [kWh]', xlabel='timestamp')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Consumption in building 1 for test data", fontsize=20)

        plt.show()

    def predict_n_ahead(self, data_input, n_ahead: int):
        dates = pd.date_range(self.data.index[-1-self.lag], periods = n_ahead+self.lag, freq='15T')[1:]
        test = pd.DataFrame(index=dates)
        #print(test.columns.values)
        test = self.preprocess(test)
        #print(test.columns.values)
        y = test[self.holi_var].tolist()
        y_holiday = test[self.holi_var].tolist()
        y_hour_cos = test[self.hour_var_cos].tolist()
        y_hour_sin = test[self.hour_var_sin].tolist()
        y_weekday_cos = test[self.day_week_cos].tolist()
        y_weekday_sin = test[self.day_week_sin].tolist()
        y_day_year_cos = test[self.day_year_cos].tolist()
        y_day_year_sin = test[self.day_year_sin].tolist()
        y_minute_cos = test[self.minutes_cos].tolist()
        y_minute_sin = test[self.minutes_sin].tolist()
        yhat = []
        X, _ = deep_learner.create_X_Y(y, y_holiday, y_hour_cos, y_hour_sin, y_weekday_cos, y_weekday_sin, y_day_year_cos, y_day_year_sin, y_minute_cos, y_minute_sin, self.lag, self.lag2)
        #print(X.shape)
        if hasattr('self', 'model') == False:
            self.load_model()
        #print(self.model)
        yhat = [y[0] for y in self.model.predict(X)]
        return yhat

    def evaluate_n_ahead(self, n_ahead: int):
        dates = pd.date_range(self.data_user.index[-1], periods = n_ahead, freq='15T')[1:]
        test = pd.DataFrame(self.predict_n_ahead(holidata_user, n_ahead))
        #test.index = holidata_user.index.union(dates)
        test.index = dates
        test.index = pd.to_datetime(test.index)
        test.to_csv(self.export_file_path, index=True)
        plt.figure(figsize=(25, 10))
        plt.grid()
        plt.gca().set(ylabel='Consumption [kWh]', xlabel='timestamp')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Consumption forecast in building 1 for given days ahead", fontsize=20)
        #plt.plot(holidata[-672*4:-672*2].index, holidata[-672*4:-672*2].loc[:,"Valeur"], color='orange', label='test', alpha=0.7)
        plt.plot(self.data_user.index, self.data_user.loc[:,"Valeur"], color='b', label='user input data', alpha=0.5)
        plt.plot(test.index, test.iloc[:,0], color='black', linestyle='--', linewidth=3, label='Forecaster model',alpha=0.7)
        plt.legend(prop={'size': 20})

deep_learner = DeepModelTS(
# USER INPUT SETTINGS
Y_var = 'Valeur',
model_load = "model_B!_complete.json",
import_file_path = './building1_input.csv',
export_file_path = './predictions.csv',
# TRAINING SETTINGS
data_path = './Consumption_15min.csv',
model_save = "model_B!_complete.json",
lag = 0,
lag2 = 672,
LSTM_layer_depth = 50,
epochs = 500,
batch_size = 128,
train_test_split = 0.05,
n_test = 3360*2
)
