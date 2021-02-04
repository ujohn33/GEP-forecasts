# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:21:52 2020

@author: afelice
"""

import pandas as pd

building_1 = pd.read_pickle('building_1_full.pkl')
building_2 = pd.read_pickle('building_2_full.pkl')
building_3 = pd.read_pickle('building_3_full.pkl')
building_4 = pd.read_pickle('building_4_full.pkl')

building_info = pd.read_excel('GEP company buildings - data info.xlsx', 
                              header=0, index_col=0)

building_info = building_info[['Company name', 'Heating', 
                               'Yearly electr consumption (kWh)', 
                               'Ground surface building (m2)', '# floors']]
building_info['Total surface'] = building_info['Ground surface building (m2)']\
    * building_info['# floors']
    
building_info = building_info.loc[building_info.index.dropna()]
building_info = building_info.drop(building_info.index[21])
building_info['Ratio'] = building_info['Yearly electr consumption (kWh)']/\
    building_info['Total surface']

categories = [3, 2, 3, 2, 2, 2, 4, 2, 2, 1, 1, 1, 2, 4, 4, 4, 2, 2, 3, 4, 2]

building_info['Category'] = categories
building_info.index = building_info.index.astype(int)

gep_full_dataset = pd.DataFrame(columns=building_info['Company name'])

for index in building_info.index:
    if building_info.iloc[index - 1]['Category'] == 1:
        gep_full_dataset.iloc[:, index - 1] = building_1['% per timestep'] * \
            building_info.iloc[index - 1]['Yearly electr consumption (kWh)']/100
            
    elif building_info.iloc[index - 1]['Category'] == 2:
        gep_full_dataset.iloc[:, index - 1] = building_2['% per timestep'] * \
            building_info.iloc[index - 1]['Yearly electr consumption (kWh)']/100
            
    elif building_info.iloc[index - 1]['Category'] == 3:
        gep_full_dataset.iloc[:, index - 1] = building_3['% per timestep'] * \
            building_info.iloc[index - 1]['Yearly electr consumption (kWh)']/100
            
    elif building_info.iloc[index - 1]['Category'] == 4:
        gep_full_dataset.iloc[:, index - 1] = building_4['% per timestep'].values * \
            building_info.iloc[index - 1]['Yearly electr consumption (kWh)']/100       

gep_full_dataset.to_pickle('gep_full_dataset_kWh.pkl')