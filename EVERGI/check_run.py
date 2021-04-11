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

def train(inputs, outputs, HORIZON=72):
    l = 1
    depth = 32
    dropout = 0.1
    patience = 10
    EPOCHS = 100
    BATCHSIZE = 1500
    
    
    if l==1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(depth, input_shape=(HORIZON, 14)),
            tf.keras.layers.Dense(HORIZON)
        ])
    elif l==2:
        model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(depth, input_shape=(HORIZON, 14), return_sequences=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(depth),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(HORIZON)
        ])

    # Compile model
    model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mape'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)

    FULL_LSTMIMO = model.fit(inputs, outputs, batch_size=BATCHSIZE, epochs=EPOCHS, validation_split=0.15, callbacks=[early_stopping])
    metrics = pd.DataFrame(columns=['mae','mape', 'rmse', 'B'], index=range(28))
    for i,df in enumerate(datasets):
        concat_input = tf.concat([dX_test[i]['X'],dX_test[i]['X2']], axis=2)
        FD_predictions = FULL_LSTMIMO.predict(concat_input)
        FD_eval_df = create_evaluation_df(FD_predictions, dX_test[i], HORIZON, dX_scaler[i])
        mae = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'MAE')
        mape = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'MAPE')
        rmse = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'RMSE')
        #print('rmse {}'.format(rmse))
        metrics.loc[i] = pd.Series({'mae':mae, 'mape':mape, 'rmse':rmse, 'B': names[i]})
    wandb.log({"mape": metrics.mape.mean()})
    wandb.log({"rmse": metrics.rmse.mean()})
    wandb.log({"mae": metrics.mae.mean()})

    
if __name__ == '__main__':
    # FETCH THE DATASETS
    #dset = 'GEP'
    #net = 'stlf'
    HORIZON = 72
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
        dX_train.append(tf.concat([train_inputs['X'],train_inputs['X2']], axis=2))
        dT_train.append(train_inputs['target'])
        dX_test.append(test_inputs)
        dX_scaler.append(y_scaler)
    global_inputs_X = tf.concat(dX_train, 0)
    global_inputs_T = tf.concat(dT_train, 0)
    
    print(global_inputs_X.shape)
    model = train(inputs=global_inputs_X, outputs=global_inputs_T)