# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:46:21 2020

@author: givkriek
"""

import pandas as pd
import numpy as np
import datetime as dt

def csvToArray(file):
    '''
    Read data from a csv file and convert into a numpy array.
    The csv file contain three columns: date, time and values
    '''
    
    dataset = pd.read_csv(file, header=None, sep='\t')
    data = dataset[:][0]
    return data


a = csvToArray('541448810000124071_01012017_31122017.csv')
b = csvToArray('541448810000124071_01012018_31122018.csv')
c = csvToArray('541448810000124071_01012019_31122019.csv')

dataset = {'Date':[],'Energy':[]}

for timestep in range(13,35052+1):
    # Add date
    day_str = a[timestep][2:18]
    day = dt.datetime.strptime(day_str,"%d/%m/%Y %H:%M")
    dataset['Date'].append(day)
    # Find kWh value
    next_coma_pos = a[timestep][19:].find(';')
    energy =  float(a[timestep][19:19+next_coma_pos].replace(',','.'))
    # Add kW value
    dataset['Energy'].append(energy)

for timestep in range(13,35052+1):
    # Add date
    day_str = b[timestep][2:18]
    day = dt.datetime.strptime(day_str,"%d/%m/%Y %H:%M")
    dataset['Date'].append(day)
    # Find kWh value
    next_coma_pos = b[timestep][19:].find(';')
    energy =  float(b[timestep][19:19+next_coma_pos].replace(',','.'))
    # Add kW value
    dataset['Energy'].append(energy)
    
for timestep in range(13,365*96+13):
    # Add date
    day_str = c[timestep][2:18]
    day = dt.datetime.strptime(day_str,"%d/%m/%Y %H:%M")
    dataset['Date'].append(day)
    # Find kWh value
    next_coma_pos = c[timestep][19:].find(';')
    energy =  float(c[timestep][19:19+next_coma_pos].replace(',','.'))
    # Add kW value
    dataset['Energy'].append(energy)    
    
np.save('dic_2017_2019.npy', dataset) 