import datetime
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler

import src.preprocessing_3days
from src.preprocessing_3days import series_to_supervised, preprocess
from src.functions import load_data, TimeSeriesTensor, create_evaluation_df, plot_train_history, validation, save_model, load_model

HORIZON = 72

def train_test_split(df, n_test):
    if len(df) < 8760:
        n_test = round(len(df) * 0.2)
    test_df = df.copy()[-(n_test+71):]
    train_df = df.copy()[:-(len(test_df)-71)]
    return train_df, test_df

def MIMO_fulldata_preparation(df, n_test=4380, T=72, HORIZON=72):
    df = df.merge(series_to_supervised(df), how='right', left_index=True, right_index=True)
    df = preprocess(df, 'Belgium')
    train_df, test_df = train_test_split(df, n_test)
    y_scaler = MinMaxScaler()
    y_scaler.fit(train_df[['value']])
    long_scaler = MinMaxScaler()
    test_df[test_df.columns] = long_scaler.fit_transform(test_df)
    train_df[train_df.columns] = long_scaler.fit_transform(train_df)
    tensor_structure = {'X':(range(-T+1, 1), train_df.columns[:1]), 'X2':(range(1, 73), train_df.columns[1:])}
    train_inputs = TimeSeriesTensor(train_df, 'value', HORIZON, tensor_structure)
    test_inputs = TimeSeriesTensor(test_df, 'value', HORIZON, tensor_structure)
    return train_inputs, test_inputs, y_scaler

class MyTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1500, 2000, step=100)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the BayesianOptimization Tuner.
        
def build_model(hp):
    l = hp.Int('layers',1,2)
    drop = hp.Float('dropout',min_value=0.1,max_value=0.3,step=0.1)
    n = hp.Int('neurons',32,256,step=32)
    if l==1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(n, input_shape=(HORIZON, 14)),
            tf.keras.layers.Dense(HORIZON)
        ])
    elif l==2:
        model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(n, input_shape=(HORIZON, 14), return_sequences=True),
            tf.keras.layers.Dropout(drop),
            tf.keras.layers.LSTM(n),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(HORIZON)
        ])

    # Compile model
    model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mape'])
    return model
    
if __name__ == '__main__':
    # FETCH THE DATASETS
    GEP1 = pd.read_csv('../data/GEP/Consumption_1H.csv', index_col=0, header=0, names=['value'])
    GEP4 = pd.read_csv('../data/GEP/B4_Consumption_1H.csv', index_col=0, header=0, names=['value'])
    datasets = [GEP1, GEP4]
    names = ['GEP1', 'GEP4']
    
    dX_train = []
    dT_train = []
    dX_test = []
    dX_scaler = []
    for i,df in enumerate(datasets):
        train_inputs, test_inputs, y_scaler = MIMO_fulldata_preparation(df, n_test=4380, T=HORIZON, HORIZON=HORIZON)
        dX_train.append(train_inputs['X'])
        dT_train.append(train_inputs['target'])
        dX_test.append(test_inputs)
        dX_scaler.append(y_scaler)
    global_inputs_X = tf.concat(dX_train, 0)
    global_inputs_T = tf.concat(dT_train, 0)
    
    working = pname+'models/'+dset+'_models/'+building
    tuner = MyTuner(build_model,objective=Objective('val_mse',direction='min'),max_trials=60,num_initial_points=4,directory=working,project_name=net+'_trials',overwrite=True)
    tuner.search(scaled_x,scaled_y,epochs=100,validation_data=(val_x,val_y),callbacks=[es],verbose=0)
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print('Best HPs for'+net,':',best_hps.values)
    # Fit best model
    model = tuner.hypermodel.build(best_hps)
