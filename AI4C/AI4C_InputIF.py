# numpy version 1.16.3
# pandas version 0.24.2
import numpy as np
import pandas as pd
import os

'''
This program aims to achieve the following:
- Incorporate the input data with the location profile feature data so that
  the resulting input data (output data) can be used to forecast the future 
  demand using the provided forecasting model.

The input file for this program is:
- quarterly_profile.csv: the profile data for each location as the mean and
                         standard deviation of the demand, and the number of 
                         non-zero demand at each weekday and time.
- dchange_quarterly_profile.csv: the profile data for each location as the 
                                 mean and standard deviation of the demand 
                                 at each weekday and time.
- q_pop_max.csv: contain the value used to normalize the number of non-zero 
                 demand features.

The output from this program is:
- np.array: the array consisting of current demand and location profile 
            feature data. 
'''


# Function for creating features given the location profile data.
def fill_features(wd, ts, q_max, ftr_sz, quarterly_p, dc_quarterly_p):
    """
    :param wd: weekday information
    :param ts: timestamp information
    :param q_max: normalization parameter
    :param ftr_sz: feature size for the location profile used in the
                   forecasting model
    :param quarterly_p: location profile data containing demand
                        characteristic
    :param dc_quarterly_p: location profile data containing demand
                           change characteristic
    :return: location profile feature data at given weekday and time
    """
    np_lookup = np.zeros((1, ftr_sz))
    # Generate features for the demand at time less than 23:30.
    if ts < 94:
        # The first three features are the mean of the demand for the next 15,
        # 30 and 45 minutes.
        np_lookup[0, 0:3] = \
            quarterly_p['mean'][wd][ts:(ts + 3)].values
        # The following three features are the standard deviation of the
        # demand for the next 15, 30 and 45 minutes.
        np_lookup[0, 3:6] = \
            quarterly_p['std'][wd][ts:(ts + 3)].values
        # The following three features are the standard deviation of the
        # demand change for the next 15, 30 and 45 minutes.
        np_lookup[0, 6:9] = \
            dc_quarterly_p['std'][wd][ts:(ts + 3)].values
        # The following three features are the number of the non-zero demand
        # for the next 15, 30 and 45 minutes.
        np_lookup[0, 9:12] = \
            quarterly_p['n_pop'][wd][ts:(ts + 3)].values

    # Generate features for the demand at time greater than 23:30.
    if ts > 94:
        # Use the demand at the succeeding weekday.
        np_lookup[0, 1:3] = \
            quarterly_p['mean'][(wd + 1) % 7][0:2].values
        np_lookup[0, 0] = quarterly_p['mean'][wd][ts]
        np_lookup[0, 4:6] = \
            quarterly_p['std'][(wd + 1) % 7][0:2].values
        np_lookup[0, 3] = quarterly_p['std'][wd][ts]
        np_lookup[0, 7:9] = \
            dc_quarterly_p['std'][(wd + 1) % 7][0:2].values
        np_lookup[0, 6] = dc_quarterly_p['std'][wd][ts]
        np_lookup[0, 10:12] = \
            quarterly_p['n_pop'][(wd + 1) % 7][0:2].values
        np_lookup[0, 9] = quarterly_p['n_pop'][wd][ts]

    # Generate features for the demand at time equal to 23:30.
    if ts == 94:
        # Use the demand at the succeeding weekday.
        np_lookup[0, 2] = quarterly_p['mean'][(wd + 1) % 7][0]
        np_lookup[0, 0:2] = \
            quarterly_p['mean'][wd][ts:(ts + 2)].values
        np_lookup[0, 5] = quarterly_p['std'][(wd + 1) % 7][0]
        np_lookup[0, 3:5] = \
            quarterly_p['std'][wd][ts:(ts + 2)].values
        np_lookup[0, 8] = dc_quarterly_p['std'][(wd + 1) % 7][0]
        np_lookup[0, 6:8] = \
            dc_quarterly_p['std'][wd][ts:(ts + 2)].values
        np_lookup[0, 11] = quarterly_p['n_pop'][(wd + 1) % 7][0]
        np_lookup[0, 9:11] = \
            quarterly_p['n_pop'][wd][ts:(ts + 2)].values
    np_lookup[0, 9:12] /= q_max
    return np_lookup


def elaborate_input(raw_input):
    """
    :param raw_input: numpy array containing:
         geo: geohash6 information
         day: day information
         timestamp: timestamp information
         demand: demand information
    :return: input data suitable for the forecasting model
    """
    # Initialize the absolute time to be used for converting
    # time data to integer.
    abs_time = pd.to_datetime('00:00', format='%H:%M')

    # Load demand characteristics and demand change characteristic data.
    cwd = os.getcwd()
    quarterly_profile = pd.read_csv(cwd +
                                    '/Traffic data/quarterly_profile.csv',
                                    sep=',',
                                    header=[0, 1, 2],
                                    index_col=0)
    quarterly_profile.columns.set_levels(
        quarterly_profile.columns.levels[2].astype(int),
        level=2,
        inplace=True)
    dc_quarterly_profile = pd.read_csv(
        cwd + '/Traffic data/dchange_quarterly_profile.csv',
        sep=',',
        header=[0, 1, 2],
        index_col=0)
    dc_quarterly_profile.columns.set_levels(
        dc_quarterly_profile.columns.levels[2].astype(int),
        level=2,
        inplace=True)
    q_max = np.genfromtxt(cwd + '/Traffic data/q_pop_max.csv')
    ftr_sz = 12

    # Start converting input data
    translated_input = np.zeros((raw_input.shape[0], 13))
    for r in range(len(raw_input)):
        geo = raw_input[r][0]
        day = raw_input[r][1]
        timestamp = raw_input[r][2]
        demand = raw_input[r][3]
        # Translate time data into integer-based data.
        ts = ((pd.to_datetime(timestamp, format='%H:%M') -
               abs_time) // 15).to_timedelta64(). \
            astype('timedelta64[m]').astype(int)
        # Obtain the weekday data.
        # If the day in the input data start from 1, shift the day by -1
        # because the day used in location profile data start from 0.
        wd = (day - 1) % 7
        # Generate the feature data
        np_locp = fill_features(wd, ts, q_max, ftr_sz,
                                quarterly_profile[geo],
                                dc_quarterly_profile[geo])
        # Form the input data suitable for the forecasting model
        # (output data).
        translated_input[r][0] = demand
        translated_input[r][1:] = np_locp

    # Return the output data.
    return translated_input
