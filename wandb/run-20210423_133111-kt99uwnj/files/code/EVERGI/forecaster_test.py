import datetime
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model

import wandb
from wandb.keras import WandbCallback
wandb.login()

# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
#from tensorflow.python.compiler.mlcompute import mlcompute
# Select CPU device.
#mlcompute.set_mlc_device(device_name='any') # Available options are 'cpu', 'gpu', and 'any'.

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import src.preprocessing_3days
from src.preprocessing_3days import series_to_supervised, preprocess
from src.functions import load_data, TimeSeriesTensor, create_evaluation_df, plot_train_history, validation, save_model, load_model

def train_test_split(df, n_test, horizon):
    if len(df) < 8760:
        n_test = round(len(df) * 0.2)
    test_df = df.copy()[-(n_test+horizon-1):]
    train_df = df.copy()[:-(len(test_df)-horizon+1)]
    return train_df, test_df

def MIMO_fulldata_preparation(df, n_test=4380, T=72, HORIZON=72, country='Canada'):
    df = df.merge(series_to_supervised(df), how='right', left_index=True, right_index=True)
    df = preprocess(df, country)
    train_df, test_df = train_test_split(df, n_test, horizon=HORIZON)
    y_scaler = MinMaxScaler()
    y_scaler.fit(train_df[['value']])
    long_scaler = MinMaxScaler()
    test_df[test_df.columns] = long_scaler.fit_transform(test_df)
    train_df[train_df.columns] = long_scaler.fit_transform(train_df)
    tensor_structure = {'X':(range(-T+1, 1), train_df.columns[:1]), 'X2':(range(1, HORIZON+1), train_df.columns[1:])}
    train_inputs = TimeSeriesTensor(train_df, 'value', HORIZON, tensor_structure)
    test_inputs = TimeSeriesTensor(test_df, 'value', HORIZON, tensor_structure)
    return train_inputs, test_inputs, y_scaler

def build_model(l, drop, n, lr):
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
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Compile model
    model.compile(loss='mse', optimizer=opt,metrics=['mse'])
    return model

if __name__ == '__main__':
    # FETCH THE DATASETS
    tf.random.set_seed(0)
    dset = 'Columbia'
    country = 'Canada'
    HORIZON = 72

    
    net = 'stlf'
    LAYERS = 1
    DROPOUT = 0.3
    NEURONS = 64
    LR = 1e-3
    BATCHSIZE = 1500

    MAX_EPOCHS = 100
    PATIENCE = 10
    datasets = []
    names = []
    
    if dset == 'GEP':
        GEP1 = pd.read_csv('../data/GEP/Consumption_1H.csv', index_col=0, header=0, names=['value'])
        GEP4 = pd.read_csv('../data/GEP/B4_Consumption_1H.csv', index_col=0, header=0, names=['value'])
        datasets = [GEP1, GEP4]
        names = ['GEP1', 'GEP4']
    if dset == 'Columbia':
        for i in range(1,29):
            filename = '../data/Columbia_clean/Residential_'+str(i)+'.csv'
            df = pd.read_csv(filename, index_col=0)
            datasets.append(df)
            names.append('B'+str(i))
    if dset == 'London':
        hourly = pd.read_csv('../data/London_smart_meters/London_hourly_all.csv', index_col='tstp')
        for house in hourly['LCLid'].unique():
            temp = hourly.loc[hourly['LCLid'] == house]
            datasets.append(temp)
            names.append(house)
        
    dX_train = []
    dT_train = []
    dX_test = []
    dX_scaler = []
    for df in tqdm(datasets):
        train_inputs, test_inputs, y_scaler = MIMO_fulldata_preparation(df, n_test=4380, T=HORIZON, HORIZON=HORIZON, country=country)
        dX_train.append(tf.concat([train_inputs['X'],train_inputs['X2']], axis=2))
        dT_train.append(train_inputs['target'])
        dX_test.append(test_inputs)
        dX_scaler.append(y_scaler)
    global_inputs_X = tf.concat(dX_train, 0)
    global_inputs_T = tf.concat(dT_train, 0)
    print('done with data')
    working = '.models/'+dset+'_models/global/trials'

    if HORIZON == 72:
        proj_name = '3days_forcast'
        if not os.path.exists('./results/'+dset+'/global/3days'):
            os.makedirs('./results/'+dset+'/global/3days')
        if not os.path.exists('./models/'+dset+'_models'):
            os.makedirs('./models/'+dset+'_models')
    if HORIZON == 24:
        proj_name = 'dayahead'
        if not os.path.exists('./results/'+dset+'/global/dayahead'):
            os.makedirs('./results/'+dset+'/global/dayahead')
        if not os.path.exists('./models/'+dset+'_models'):
            os.makedirs('./models/'+dset+'_models')
    # 1️⃣ Start a new run, tracking config metadata
    run = wandb.init(project=proj_name, config={
        'layers': LAYERS,
        'dropout': DROPOUT,
        'neurons': NEURONS,
        'learning rate': LR,
        'batch_size': BATCHSIZE,

        "architecture": "global",
        "dataset": dset,
        "epochs": MAX_EPOCHS,
        'patience': PATIENCE
    })
    config = wandb.config

    # full data LSTM MIMO compilation and fit
    LSTMIMO = build_model(l=LAYERS, drop=DROPOUT, n=NEURONS, lr=LR)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')

    history = LSTMIMO.fit(global_inputs_X, global_inputs_T, batch_size=BATCHSIZE, epochs=MAX_EPOCHS,
                          validation_split=0.15,
                          callbacks=[early_stopping, WandbCallback()], verbose=1)

    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Fit best model
    final_model = LSTMIMO.fit(global_inputs_X,global_inputs_T,epochs=best_epoch,batch_size=BATCHSIZE,verbose=0)

    metrics = pd.DataFrame(columns=['mae','mape', 'rmse', 'B'], index=range(28))
    for i,df in enumerate(datasets):
        concat_input = tf.concat([dX_test[i]['X'],dX_test[i]['X2']], axis=2)
        FD_predictions = LSTMIMO.predict(concat_input)
        FD_eval_df = create_evaluation_df(FD_predictions, dX_test[i], HORIZON, dX_scaler[i])
        FD_eval_df.index = pd.to_datetime(FD_eval_df['timestamp'])
        FD_eval_df = FD_eval_df[np.where(FD_eval_df.index.hour == 0)[0][0]:][::24]
        FD_eval_df.to_csv('./results/'+dset+'/'+wandb.run.name+'_'+str(i)+'.csv')
        mae = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'MAE')
        mape = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'MAPE')
        rmse = validation(FD_eval_df['prediction'], FD_eval_df['actual'], 'RMSE')
        #print('rmse {}'.format(rmse))
        metrics.loc[i] = pd.Series({'mae':mae, 'mape':mape, 'rmse':rmse, 'B': names[i]})
    wandb.log({"mape": metrics.mape.mean()})
    wandb.log({"rmse": metrics.rmse.mean()})
    wandb.log({"mae": metrics.mae.mean()})
    if HORIZON == 72:
        metrics.to_csv('./results/'+dset+'/global/3days/LSTM_'+wandb.run.name+'.csv')
        model_path = '.models/'+dset+'_models/global_'+wandb.run.name
        save_model(LSTMIMO, model_path)
    if HORIZON == 24:
        metrics.to_csv('./results/'+dset+'/global/dayahead/LSTM_'+wandb.run.name+'.csv')
        model_path = './models/'+dset+'_models/global_'+wandb.run.name
        save_model(LSTMIMO, model_path)
    run.finish()
