# numpy version 1.16.3
# pandas version 0.24.2
# h5py version 2.9.0
import numpy as np
import pandas as pd
import os
import h5py
import multiprocessing as mp

'''
This program aims to achieve the following:
- Generate the training data complete with the engineered features to be used
  in training the forecasting model.
  The features are the characteristic (mean, standard deviation and number of 
  non-zero) of the future demand and future demand change for the next 15, 30 
  and 45 minutes.

The input files for this program are:
- geo_set.csv: list of unique location represented by the geohash6, latitude
               and longitude
- training.csv: training data containing geohash6, demand, day and timestamp 
                data.
- quarterly_profile.csv: the profile data for each location as the mean and
                         standard deviation of the demand, and the number of 
                         non-zero demand at each weekday and time.
- dchange_quarterly_profile.csv: the profile data for each location as the 
                                 mean and standard deviation of the demand 
                                 at each weekday and time.

The output files from this program are:
- final_fullset.h5: the final training data to be used in training the 
                    forecasting model.
- q_pop_max.csv: the value used to normalize number non-zero demands features.

Note :
- The input training data must be located within the folder 'Traffic data' 
  within the same directory and named as 'training.csv'. 
- This program employs multiprocessing.
- This program attempts to write a hdf5 file.
'''

# Get working directory.
cwd = os.getcwd()

# Load geo location data.
print("Load geo location data")
geo_df = pd.read_csv(cwd + '/Traffic data/geo_set.csv',
                     sep=',',
                     index_col=0)

# Load training data with future demand.
print("Load pre-final training data")
tr_df = pd.read_csv(cwd + '/Traffic data/prefinal_training.csv',
                    sep=',',
                    index_col=0)

# Load demand characteristics and demand change characteristic data.
print("Load demand characteristic data")
quarterly_profile = pd.read_csv(cwd + '/Traffic data/quarterly_profile.csv',
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

# Create template lookup dataframe.
# The lookup dataframe will contain the list of features for each location at
# each weekday that will be used to train the forecasting
# model.
# The lookup dataframe is intended to speed up the creation the final
# training data for the forecasting model by providing a lookup table, so that
# the features can be copied from the lookup table and joined with the demand
# for the specific location, weekday and time, without having to re-construct
# the features for each demand data.
p_lookup_df = pd.DataFrame(
    data=np.transpose([np.arange(96 * 7).astype(int) for i in range(3)]),
    columns={'weekday', 'hour', 'timestamp'},
    index=np.arange(96 * 7).astype(int))
p_lookup_df['weekday'] = (p_lookup_df['weekday'] // 96) % 7
p_lookup_df['hour'] = (p_lookup_df['hour'] // 4) % 24
p_lookup_df['timestamp'] = p_lookup_df['timestamp'] % 96


# Function for creating the lookup table given the location profile data.
def fill_lookup(ftr_sz, p_lookup, quarterly_p, dc_quarterly_p):
    """
    :param ftr_sz: feature size for the location profile used in the
                   forecasting model
    :param p_lookup: template for the lookup table array
    :param quarterly_p: location profile data containing demand characteristic
    :param dc_quarterly_p: location profile data containing demand
                           change characteristic
    :return: lookup table array given the location profile
    """
    np_lookup = np.zeros((96 * 7, ftr_sz))
    # Generate features for the demand at time less than 23:30.
    filter_lookup_df = p_lookup.loc[p_lookup['timestamp'] < 94]
    for x in list(zip(filter_lookup_df.index,
                      filter_lookup_df['weekday'],
                      filter_lookup_df['hour'],
                      filter_lookup_df['timestamp'])):
        # The first three features are the mean of the demand for the next 15,
        # 30 and 45 minutes.
        np_lookup[x[0], 0:3] = \
            quarterly_p['mean'][x[1]][(x[3]):(x[3] + 3)].values
        # The following three features are the standard deviation of the
        # demand for the next 15, 30 and 45 minutes.
        np_lookup[x[0], 3:6] = \
            quarterly_p['std'][x[1]][(x[3]):(x[3] + 3)].values
        # The following three features are the standard deviation of the
        # demand change for the next 15, 30 and 45 minutes.
        np_lookup[x[0], 6:9] = \
            dc_quarterly_p['std'][x[1]][(x[3]):(x[3] + 3)].values
        # The following three features are the number of the non-zero demand
        # for the next 15, 30 and 45 minutes.
        np_lookup[x[0], 9:12] = \
            quarterly_p['n_pop'][x[1]][(x[3]):(x[3] + 3)].values

    # Generate features for the demand at time greater than 23:30.
    filter_lookup_df = p_lookup.loc[(p_lookup['timestamp'] > 94)]
    for x in list(zip(filter_lookup_df.index,
                      filter_lookup_df['weekday'],
                      filter_lookup_df['timestamp'])):
        # Use the demand at the succeeding weekday.
        np_lookup[x[0], 1:3] = \
            quarterly_p['mean'][(x[1] + 1) % 7][0:2].values
        np_lookup[x[0], 0] = quarterly_p['mean'][x[1]][x[2]]
        np_lookup[x[0], 4:6] = \
            quarterly_p['std'][(x[1] + 1) % 7][0:2].values
        np_lookup[x[0], 3] = quarterly_p['std'][x[1]][x[2]]
        np_lookup[x[0], 7:9] = \
            dc_quarterly_p['std'][(x[1] + 1) % 7][0:2].values
        np_lookup[x[0], 6] = dc_quarterly_p['std'][x[1]][x[2]]
        np_lookup[x[0], 10:12] = \
            quarterly_p['n_pop'][(x[1] + 1) % 7][0:2].values
        np_lookup[x[0], 9] = quarterly_p['n_pop'][x[1]][x[2]]

    # Generate features for the demand at time equal to 23:30.
    filter_lookup_df = p_lookup.loc[(p_lookup['timestamp'] == 94)]
    for x in list(zip(filter_lookup_df.index,
                      filter_lookup_df['weekday'],
                      filter_lookup_df['timestamp'])):
        # Use the demand at the succeeding weekday.
        np_lookup[x[0], 2] = quarterly_p['mean'][(x[1] + 1) % 7][0]
        np_lookup[x[0], 0:2] = \
            quarterly_p['mean'][x[1]][(x[2]):(x[2] + 2)].values
        np_lookup[x[0], 5] = quarterly_p['std'][(x[1] + 1) % 7][0]
        np_lookup[x[0], 3:5] = \
            quarterly_p['std'][x[1]][(x[2]):(x[2] + 2)].values
        np_lookup[x[0], 8] = dc_quarterly_p['std'][(x[1] + 1) % 7][0]
        np_lookup[x[0], 6:8] = \
            dc_quarterly_p['std'][x[1]][(x[2]):(x[2] + 2)].values
        np_lookup[x[0], 11] = quarterly_p['n_pop'][(x[1] + 1) % 7][0]
        np_lookup[x[0], 9:11] = \
            quarterly_p['n_pop'][x[1]][(x[2]):(x[2] + 2)].values

    return np_lookup


def gen_train(ftr_size, partial_df, p_lookup, quarter_p, dc_quarter_p):
    """
    :param ftr_size: feature size for the location profile used in the
                     forecasting model
    :param partial_df: training data at a specific location
    :param p_lookup: template for the lookup dataframe
    :param quarter_p: location profile data containing demand characteristic
    :param dc_quarter_p: location profile data containing demand
                         change characteristic
    :return: partial input data (limited to specific location) for the
             forecasting model
    """
    # Define the lookup key based on the weekday and timestamp.
    partial_df.insert(partial_df.shape[1], 'lookup', 0)
    partial_df['lookup'] = partial_df['time'] % (96 * 7)
    partial_df.reset_index(drop=True, inplace=True)

    # Create an empty set to be filled with location profile features data.
    partial_xset = np.zeros((partial_df.shape[0], ftr_size))
    # Create the lookup table as np.array.
    template_np = fill_lookup(ftr_size, p_lookup,
                              quarter_p, dc_quarter_p)

    # Generate the location profile feature data for each demand based on
    # the lookup key of each demand.
    for x in list(zip(partial_df.index,
                      partial_df['lookup'])):
        partial_xset[x[0], :] = template_np[x[1], :]

    # Combine the demand, location profile feature and future demand data
    # into single np.array.
    final_ar = np.concatenate((np.reshape(partial_df['demand'].values,
                                          (-1, 1)),
                               np.round(partial_xset, 5),
                               np.reshape(partial_df['t_demand'].values,
                                          (-1, 1))),
                              axis=1)
    return final_ar


print("Generate final training set")
# Set the feature size to 12, explanation of each feature is in fill_lookup
# function.
feature_size = 12
geo = 0
# Function to show the progress of final training data construction.
def mp_proggress(args):
    global geo
    geo += 1
    print('\r%s |%s| %s%%' % ("Generating final training data: ",
                              '#' * int(geo / geo_df.shape[0] * 20) +
                              '_' * int((geo_df.shape[0] - geo) /
                                        geo_df.shape[0] * 20),
                              "{0:.2f}". format(geo / geo_df.shape[0] * 100)),
          end='')


# Use multiprocessing while generating the final training set to save time.
pool = mp.Pool(processes=mp.cpu_count())
procs = [pool.apply_async(
    gen_train,
    args=(feature_size,
          tr_df.loc[tr_df['geohash6'] == temp_geo, :],
          p_lookup_df,
          quarterly_profile[temp_geo],
          dc_quarterly_profile[temp_geo]),
    callback=mp_proggress)
    for temp_geo in geo_df['geohash6']]

pool.close()
pool.join()

# Construct the final training data obtained from multiprocessing as np.array.
fullset = np.concatenate([procs[p].get() for p in range(len(procs))],
                         axis=0)
print()

# Clean any NaN data.
fullset = fullset[~np.isnan(fullset).any(axis=1)]

# Normalize the number of non-zero demand (last three location profile
# features) to the range of 0~1.
q_max = np.amax(fullset[:, (feature_size - 2):(feature_size + 1)],
                initial=1)
fullset[:, (feature_size - 2):(feature_size + 1)] /= q_max
# Resulting training set ready to be used to train the forecasting model.
'''
[[0.05485798 0.1816     0.19892    ... 1.         1.         0.08620924]
 [0.08620924 0.19892    0.19525    ... 1.         1.         0.05073921]
 [0.05073921 0.19525    0.17755    ... 1.         1.         0.0751742 ]
 ...
 [0.00219915 0.0022     0.         ... 0.         0.         0.00606053]
 [0.00222021 0.00222    0.         ... 0.         0.         0.00166894]
 [0.00306231 0.00306    0.         ... 0.         0.         0.02934427]]
'''

# Save the value used to normalize the last three location profile features
# to csv file to be used when forecasting the demand.
q_max = [q_max]
np.savetxt(cwd + '/Traffic data/q_pop_max.csv', q_max)

# Save the final training set in hdf5 file.
hf = h5py.File(cwd + '/Traffic data/final_fullset.h5', 'w')
hf.create_dataset('fullset', data=fullset)
hf.close()
