# numpy version 1.16.3
# pandas version 0.24.2
import numpy as np
import pandas as pd
import os
import multiprocessing as mp

'''
This program aims to achieve the following:
- Generate the location profile which characterize the demand pattern
  at certain location.
  
The input files for this program are:
- geo_set.csv: list of unique location represented by the geohash6, latitude
               and longitude
- time_set.csv: list of unique time-key data represented by day in integer,
                timestamp in HH:mm format and time as unique time-key.
- complete_data.csv: demand data at each location and each time-key.

The output files from this program are:
- quarterly_profile.csv: the profile data for each location as the mean and
                         standard deviation of the demand, and the number of 
                         non-zero demand at each weekday and time.
- dchange_quarterly_profile.csv: the profile data for each location as the 
                                 mean and standard deviation of the demand 
                                 at each weekday and time.
                                 
Note :
- The input training data must be located within the folder 'Traffic data' 
  within the same directory and named as 'training.csv'. 
- This program employs multiprocessing.
'''

# Get working directory.
cwd = os.getcwd()

# Load geo-location data as pandas dataframe.
print("Load geo location data")
geo_df = pd.read_csv(cwd + '/Traffic data/geo_set.csv',
                     sep=',',
                     index_col=0)

# Load time-key list as pandas dataframe.
print("Load time list")
time_df = pd.read_csv(cwd + '/Traffic data/time_set.csv',
                      sep=',',
                      index_col=0)
time_span = time_df.shape[0]

# Load 'complete' data containing the demand information at each location
# and unique time.
print("Load complete data")
ori_complete_df = pd.read_csv(cwd + '/Traffic data/complete_data.csv',
                              sep=',')


# Function for profiling the demand characteristic at certain location
# during certain weekday (sun, mon, ...) as represented by the mean, standard
# deviation and the number of non-zero demand.
def meansd_profile(temp_geo, temp_df, pfl_df):
    """
    :param temp_geo: location information as geohash6
    :param temp_df: training data only for the specified location
    :param pfl_df: empty template for profile dataframe
    :return: profile dataframe for the specified location
    """
    # Profile the demand for each weekday because weekday affects demand.
    for g in range(7):
        pfl_df[temp_geo]['mean'][g] = temp_df.loc[
            (temp_df['day'] == g) & (temp_df[temp_geo] > 0)].groupby(
            ['timestamp'])[temp_geo].agg(np.mean)
        pfl_df[temp_geo]['std'][g] = temp_df.loc[
            (temp_df['day'] == g) & (temp_df[temp_geo] > 0)].groupby(
            ['timestamp'])[temp_geo].agg(np.std)
        pfl_df[temp_geo]['n_pop'][g] = temp_df.loc[
            (temp_df['day'] == g) & (temp_df[temp_geo] > 0)].groupby(
            ['timestamp'])[temp_geo].agg(np.count_nonzero)
    pfl_df.fillna(value=0.0, inplace=True)
    return pfl_df


# Function for profiling the demand change characteristic at certain location
# during certain weekday (sun, mon, ...) as represented by the mean and
# standard deviation of the demand change (non-zero column is not used).
def dc_meansd_profile(temp_geo, temp_df, pfl_df):
    """
    :param temp_geo: location information as geohash6
    :param temp_df: training data only for the specified location
    :param pfl_df: empty template for profile dataframe
    :return: profile dataframe for the specified location
    """
    # Calculate the change in demand by comparing two subsequent demands.
    # The demand change is the difference between current demand and the next
    # 15 minutes demand.
    temp_df['delta_d'][temp_df.shape[0] - 1] = 0
    for x in list(zip(temp_df['delta_d'][:-1].index,
                      temp_df[temp_geo][1:],
                      temp_df[temp_geo][:-1])):
        temp_df['delta_d'][x[0]] = x[1] - x[2]
    # Profile the demand change for each weekday because weekday affects
    # demand change.
    for g in range(7):
        pfl_df[temp_geo]['mean'][g] = temp_df.loc[
            (temp_df['day'] == g) & (temp_df[temp_geo] >= 0)].groupby(
            ['timestamp'])['delta_d'].agg(np.mean)
        pfl_df[temp_geo]['std'][g] = temp_df.loc[
            (temp_df['day'] == g) & (temp_df[temp_geo] >= 0)].groupby(
            ['timestamp'])['delta_d'].agg(np.std)
        # pfl_df[temp_geo]['n_pop'][g] = temp_df.loc[
        #     (temp_df['day'] == g) & (temp_df[temp_geo] > 0)].groupby(
        #     ['timestamp'])['delta_d'].agg(np.count_nonzero)
    pfl_df.fillna(value=0.0, inplace=True)
    return pfl_df


# Weekday matters! (monday and sunday are different days after all)
# Profiling the demand characteristic at each location and during each weekday.
print("Profiling each location using mean and sd of demands on each weekday")
# Use quarter-hour resolution (per 15 minutes) instead of hour resolution
# (per hour, h_period = 1).
h_period = 4
profile_name = 'quarterly_profile'
complete_df = ori_complete_df.copy()
# Constructing data structure template.
profile_df = pd.DataFrame(data=[[0.0 for j in range(7)]
                                for i in range(24 * h_period)],
                          columns=[i for i in range(7)])
profile_df = pd.concat([profile_df for i in range(3)],
                       axis=1,
                       keys=['mean', 'std', 'n_pop'])
profile_df = pd.concat([profile_df
                        for i in range(geo_df.shape[0])],
                       axis=1,
                       keys=geo_df['geohash6'])

'''
geohash6 qp03wc                           ... qp0d45                         
           mean                           ...  n_pop                         
              0    1    2    3    4    5  ...      1    2    3    4    5    6
0           0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0
1           0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0
2           0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0
...
'''
# From now on, the day start from 0 instead of 1 (as in training data).
# Regardless, there are still 7 unique weekdays.
# Incorporate weekday and timestamp information into 'complete' data.
ori_temp_df = complete_df.copy()
ori_temp_df = ori_temp_df.join(
    pd.DataFrame(data=np.transpose([np.arange(ori_temp_df.shape[0]),
                                    np.arange(ori_temp_df.shape[0])]),
                 columns={'timestamp', 'day'},
                 index=np.arange(ori_temp_df.shape[0]).astype(int)))
ori_temp_df['timestamp'] = ori_temp_df['timestamp'] % (24 * h_period)
ori_temp_df['day'] = ori_temp_df['day'] // (24 * h_period) % 7

# Function for reporting the progress of profiling process.
geo = 0
def mp_proggress(args):
    global geo
    geo += 1
    print('\r%s |%s| %s%%' % ("Building profile data: ",
                              '#' * int(geo / geo_df.shape[0] * 20) +
                              '_' * int((geo_df.shape[0] - geo) /
                                        geo_df.shape[0] * 20),
                              "{0:.2f}".format(geo / geo_df.shape[0] * 100)),
          end='')


# Use multiprocessing for profiling process to save time.
pool = mp.Pool(processes=mp.cpu_count())
procs = [pool.apply_async(
    meansd_profile,
    args=(temp_geo,
          ori_temp_df[[temp_geo, 'day', 'timestamp']],
          profile_df[[temp_geo]]),
    callback=mp_proggress)
    for temp_geo in geo_df['geohash6']]

pool.close()
pool.join()

# Construct the demand profile obtained from multiprocessing as pandas
# dataframe.
final_profile_df = pd.DataFrame()
for p in range(len(procs)):
    final_profile_df = pd.concat((final_profile_df, procs[p].get()),
                                 axis=1)

# Save the demand profile data as csv file.
final_profile_df.to_csv(cwd + '/Traffic data/' + profile_name + '.csv',
                        sep=',')
print()

# From exactly one quarter to the next, there is a 'potential' pattern
# in demand changes
# (demand 'might' go up or down on particular period but the data still
# requires adjustment to absorb the noises).
# The changes are more visible in per hourly period, but data with hourly
# period useful (IMO).
# Profiling the demand change characteristic at each location and during
# each weekday as the mean and standard deviation of change in demand change.
print("Profiling each location using mean and sd of change in "
      "demands on each weekday")
# reuse the data structure used in demand profile.
profile_df[:] = 0

# Incorporate weekday and timestamp information into 'complete' data.
ori_temp_df = complete_df.copy()
ori_temp_df = ori_temp_df.join(
    pd.DataFrame(data=np.transpose([np.arange(ori_temp_df.shape[0]). \
                                   astype(float),
                                    np.arange(ori_temp_df.shape[0]),
                                    np.arange(ori_temp_df.shape[0])]),
                 columns={'delta_d', 'timestamp', 'day'},
                 index=np.arange(ori_temp_df.shape[0]).astype(int)))
ori_temp_df['timestamp'] = ori_temp_df['timestamp'] % (24 * h_period)
ori_temp_df['day'] = ori_temp_df['day'] // (24 * h_period) % 7

# Use multiprocessing for profiling process to save time.
geo = 0
pool = mp.Pool(processes=mp.cpu_count())
procs = [pool.apply_async(
    dc_meansd_profile,
    args=(temp_geo,
          ori_temp_df[[temp_geo, 'delta_d', 'day', 'timestamp']],
          profile_df[[temp_geo]]),
    callback=mp_proggress)
    for temp_geo in geo_df['geohash6']]

pool.close()
pool.join()

# Construct the demand change profile obtained from multiprocessing as pandas
# dataframe.
final_profile_df = pd.DataFrame()
for p in range(len(procs)):
    final_profile_df = pd.concat((final_profile_df, procs[p].get()),
                                 axis=1)

# Save the demand change profile data as csv file.
final_profile_df.to_csv(cwd + '/Traffic data/dchange_' + profile_name + '.csv',
                        sep=',')
print()
