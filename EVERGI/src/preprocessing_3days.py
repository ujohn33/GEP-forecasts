import pandas as pd
import numpy as np
from workalendar.usa import Texas
from workalendar.europe import Belgium, UnitedKingdom
from workalendar.usa import UnitedStates
from workalendar.america import Canada



DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
DAYS_OF_WEEK = ['week_1','week_2','week_3','week_4','week_5','week_6','week_7']
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
MINUTES_IN_DAY = MINUTES_IN_HOUR * HOURS_IN_DAY

def get_fractional_hour_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional hour in range 0-24, e.g. 12h30m --> 12.5.
    Accurate to 1 minute.
    """
    hour = series.hour
    minute = series.minute
    return hour + minute / MINUTES_IN_HOUR

def get_fractional_day_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional day in range 0-1, e.g. 12h30m --> 0.521.
    Accurate to 1 minute
    """
    fractional_hours = get_fractional_hour_from_series(series)
    return fractional_hours / HOURS_IN_DAY

def get_fractional_year_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional year in range 0-1.
    Accurate to 1 day
    """
    return (series.dayofyear - 1) / DAYS_IN_YEAR

def normalize(self, tensor):
    self.scaler = MinMaxScaler(feature_range=(0, 1))
    tensor = self.scaler.fit_transform(tensor)
    return tensor

# convert series to supervised learning
def series_to_supervised(data, dropnan=True, lag2=168):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    cols.append(df.iloc[:,0].shift(lag2))
    names += [('value'+'(t-%d)' % (lag2))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg 

def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index[::24]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target'][::24]).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df

def preprocess(dataframe, country: str):
    dataframe.index = pd.to_datetime(dataframe.index)
    # Removing duplicates
    dataframe = dataframe[~dataframe.index.duplicated()]
    
    # resample to 1 H granularity
    dataframe = dataframe.resample('H').sum()
    
    #Filling NaN values
    dataframe = dataframe.interpolate()

    # Setting the calendar holiday dates
    if country in ['Belgium', 'belgium']:
        cal = Belgium()
    elif country in ['England', 'UK', 'uk', 'United Kingdom']:
        cal = UnitedKingdom()      
    elif country in ['Texas', 'texas']:
        cal = Texas()
    elif country in ['United States', 'US', 'United States of America', 'us', 'usa', 'USA']:
        cal = UnitedStates()
    elif country in ['Canada', 'canada']:
        cal = Canada()    
    else:
        raise TypeError("No country is input to the preprocessing function") 
    
    years = list(range(2014, 2025))
    holidays = []
    for year in years:
        holidays.extend(cal.holidays(year))

    dataframe = dataframe.sort_index()

    # Rename the target column to 'Valeur' for convenience
    dataframe.rename(columns={dataframe.columns[0]: 'value'}, inplace=True)
    
    #fractional hour [0,1]
    dataframe['fractional hour'] = get_fractional_day_from_series(dataframe.index)
    
    # fractional day of year
    dataframe['day of year'] = get_fractional_year_from_series(dataframe.index)
    
    # we encode cynical data into two dimensions using a sine and cosine transformations
    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data
    dataframe = encode(dataframe, 'fractional hour', HOURS_IN_DAY)
    dataframe = encode(dataframe, 'day of year', DAYS_IN_YEAR)
    # dropping original columns
    dataframe = dataframe.drop(['fractional hour','day of year'], axis=1)

    #working day {0,1}
    dataframe['working day'] = dataframe.index.map(cal.is_working_day).astype(np.float32)
    #dataframe['working day2'] = dataframe.index.map(cal.is_working_day).astype(np.float32)
    #dataframe['working day3'] = dataframe.index.map(cal.is_working_day).astype(np.float32)


    # day of week one-hot encoded
    dataframe['day of week'] = dataframe.index.dayofweek + 1
    #dataframe['day of week2'] = (dataframe.index.dayofweek + 2) % 7
    #dataframe['day of week3'] = (dataframe.index.dayofweek + 3) % 7

    dataframe['day of week'] = pd.Categorical(dataframe['day of week'], categories=[1,2,3,4,5,6,7], ordered=True)
    #dataframe['day of week2'] = pd.Categorical(dataframe['day of week2'], categories=[1,2,3,4,5,6,7], ordered=True)
    #dataframe['day of week3'] = pd.Categorical(dataframe['day of week3'], categories=[1,2,3,4,5,6,7], ordered=True)

    dataframe = pd.get_dummies(dataframe,prefix=['week'], columns = ['day of week'], drop_first=False)
    #dataframe = pd.get_dummies(dataframe,prefix=['week2'], columns = ['day of week2'], drop_first=False)
    #dataframe = pd.get_dummies(dataframe,prefix=['week3'], columns = ['day of week3'], drop_first=False)
    #dataframe = pd.concat([dataframe, pd.DataFrame(columns=DAYS_OF_WEEK)]).fillna(0)


    return dataframe