# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:33:54 2020

@author: afelice
"""

import pandas as pd
import glob
import re
# from matplotlib import pyplot as plt

filenames = sorted(glob.glob('*.csv'))
statistics = pd.DataFrame()

building_1 = "124071"
building_2 = "124194"
building_3 = "203363"
building_4 = "283501"
building_5 = "011921"
pv = "PV"

building_1_data = pd.DataFrame()
building_2_data = pd.DataFrame()
building_3_data = pd.DataFrame()
building_4_data = pd.DataFrame()
building_5_data = pd.DataFrame()
pv_data = pd.DataFrame()
gep_data = pd.DataFrame()

def concatenate_dfs(name, file, dataset, subdata):

    if name in file:
        dataset = dataset.append(subdata)
    return dataset

for file in filenames:
    data = pd.read_csv(file, delimiter=';', header=12, decimal=',')
    data = data.loc[data['Canal'] == 'A']
    clean_data = data[['Date', 'Valeur']]
    clean_data.index = clean_data['Date']
    clean_data = clean_data[['Valeur']]
    file = re.sub('\.csv$', '', file)
    clean_data.to_pickle(file + ".pkl")
    statistics[file] = clean_data['Valeur'].describe()
    print('hi')

    building_1_data = concatenate_dfs(building_1, file, building_1_data,
                                      clean_data)
    building_2_data = concatenate_dfs(building_2, file, building_2_data,
                                      clean_data)
    building_3_data = concatenate_dfs(building_3, file, building_3_data,
                                      clean_data)
    building_4_data = concatenate_dfs(building_4, file, building_4_data,
                                      clean_data)
    building_5_data = concatenate_dfs(building_5, file, building_5_data,
                                      clean_data)
    pv_data = concatenate_dfs(pv, file, pv_data, clean_data)

gep_data['Building 1'] = building_1_data['Valeur'].values
gep_data['Building 2'] = building_2_data['Valeur'].values
gep_data['Building 3'] = building_3_data['Valeur'].values
gep_data['Building 5'] = building_5_data['Valeur'].values

building_1_data.to_pickle('building_1_cnsumptions.pkl')
building_2_data.to_pickle('building_2_cnsumptions.pkl')
building_3_data.to_pickle('building_3_cnsumptions.pkl')
building_4_data.to_pickle('building_4_cnsumptions.pkl')
building_5_data.to_pickle('building_5_cnsumptions.pkl')
pv_data.to_pickle('pv_building_5.pkl')

gep_data.to_pickle('gep_consumptions.pkl')
