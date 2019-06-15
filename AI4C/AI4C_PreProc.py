# numpy version 1.16.3
# pandas version 0.24.2
# Geohash version 1.0
import numpy as np
import pandas as pd
import Geohash as gh
import os

'''
This program aims to achieve the following:
- Generate a list of unique geo-location information and unique time list.
- Generate a 'complete' data containing the demand at each location at each time.
- Generate a modified training data that include future (next 15 minutes) 
  demand information.

The input file for this program is:
- training.csv: training data containing geohash6, demand, day and timestamp 
                data.
                 
The output files from this program are:
- geo_set.csv: list of unique location represented by the geohash6, latitude
               and longitude
- time_set.csv: list of unique time-key data represented by day in integer,
                timestamp in HH:mm format and time as unique time-key.
- complete_data.csv: demand value at each location and each time-key.
- prefinal_training.csv: training data that includes target demand.

Note :
- The input training data must be located within the folder 'Traffic data' 
  within the same directory and named as 'training.csv'. 
'''

# Initialize the absolute time to be used for converting time data to integer.
abs_time = pd.to_datetime('00:00', format='%H:%M')

# Get working directory.
cwd = os.getcwd()

# Load input training data as pandas dataframe.
print("Reading input data")
tr_df = pd.read_csv(cwd + '/Traffic data/training.csv', sep=',')

# Convert the timestamp into integer value with interval of 1 for each
# 15 minutes, therefore there are 96 unique value for a day
# (24 hour * (60 minutes / 15 minutes)).
print("Pre-process time data")
tr_df['timestamp'] = ((pd.to_datetime(tr_df['timestamp'], format='%H:%M') -
                       abs_time) // 15).astype('timedelta64[m]').astype(int)


# Set number of days considered in building the profile
num_day = 61
# Update training data consider only the demands in the last num_day
max_day = tr_df['day'].max()
if max_day - num_day >= 0:
    tr_df = tr_df.loc[tr_df['day'] > max_day - num_day]
    tr_df['day'] -= max_day - num_day

# Create a dataframe to store geo-location data information which include
# the geohash6 as the key, and latitude and longitude information.
# At the moment, this data is not used, since the location has been nicely
# clustered in a grid-like fashion thanks to the geohash precision reduction.
# However, it is necessary to cluster the locations when the locations are
# chaotically scattered on the map.
print("Pre-process geo-location data")
num_geo = tr_df['geohash6'].unique().shape[0]
geo_df = pd.DataFrame(data=np.transpose([['' for i in range(num_geo)],
                                         np.zeros(num_geo),
                                         np.zeros(num_geo)]),
                      columns=['geohash6', 'latitude', 'longitude'])

# Obtain the location information for each unique geohash6.
i = 0
for x in zip(tr_df['geohash6'].unique()):
    geo_df.loc[i]['geohash6'] = x[0]
    geo_df.loc[i]['latitude', 'longitude'] = gh.decode(x[0])
    i += 1
# Resulting dataframe of the geo-location data:
'''
     geohash6  latitude  longitude
0      qp03wc -5.353088  90.653687
1      qp03pn -5.413513  90.664673
2      qp09sw -5.325623  90.906372
3      qp0991 -5.353088  90.752563
...
1326   qp03yn -5.281677  90.620728
1327   qp09v9 -5.309143  90.950317
1328   qp0d45 -5.254211  90.796509
'''
# Save the geo-location data into csv file for easy access when necessary.
geo_df.to_csv(cwd + '/Traffic data/geo_set.csv', sep=',')

# Add time-key information representing each unique time and day.
# Basically convert the day and timestamp information into 1-D time
# information.
tr_df.insert(tr_df.shape[1], 'time', 0)
tr_df['time'] = ((tr_df['day'] - 1) * (4 * 24) +
                 tr_df['timestamp']).astype(int)

# Generate a full list of unique time-key and its respective day and timestamp
# with timestamp already in integer form
print("Pre-process time list full")
temp_time = np.arange(0, num_day * 96)
temp_hhmm = ['' for x in range(num_day * 96)]
for x in list(zip(temp_time)):
    temp_hhmm[x[0]] = str(((x[0]) // 4) % 24) + ':' + str((x[0] % 4) * 15)

temp_time_t = np.rec.fromarrays((temp_time // (4 * 24) + 1,
                                 temp_hhmm,
                                 temp_time
                                 ),
                                names=('day', 'timestamp', 'time'))
time_df = pd.DataFrame.from_records(data=temp_time_t)
time_df.sort_values(by=['time'],
                    ascending=True,
                    inplace=True)
time_df.reset_index(drop=True, inplace=True)
time_span = time_df.shape[0]
# Resulting dataframe of time-key list:
'''
      day timestamp  time
0       1       0:0     0
1       1      0:15     1
2       1      0:30     2
3       1      0:45     3
...
5853   61     23:15  5853
5854   61     23:30  5854
5855   61     23:45  5855
'''
# Save the time-key list data into csv for easy access when necessary.
time_df.to_csv(cwd + '/Traffic data/time_set.csv', sep=',')

# Generate a complete list of demand for each unique time and each location.
# Fill the missing demand data with 0.0 (or any default demand value for
# missing demand data).
print("Create complete data")
complete_df = pd.DataFrame(data=[[0.0 for j in range(geo_df.shape[0])]
                                 for i in range(time_span)],
                           columns=geo_df['geohash6'].values.tolist(),
                           index=time_df['time'].values)
complete_df.index.names = ['time']

# Projecting the demand data into the location-time 'complete' dataframe
# Can be improved by applying multiprocessing for a batch of locations
# (loss of data transfer is higher than efficiency gained from multi-processing
# when each process only processes one location at a time.
for x in list(zip(tr_df['geohash6'], tr_df['time'], tr_df['demand'])):
    complete_df[x[0]][x[1]] += x[2]
# Resulting complete dataframe:
'''
        qp03wc    qp03pn    qp09sw  ...  qp0d4m  qp03yn  qp09v9  qp0d45
time                                ...                                
0     0.054858  0.000000  0.022881  ...     0.0     0.0     0.0     0.0
1     0.086209  0.005546  0.019733  ...     0.0     0.0     0.0     0.0
2     0.050739  0.013577  0.023053  ...     0.0     0.0     0.0     0.0
3     0.075174  0.004720  0.029018  ...     0.0     0.0     0.0     0.0
...
5853  0.034809  0.000000  0.014848  ...     0.0     0.0     0.0     0.0
5854  0.059814  0.000000  0.007184  ...     0.0     0.0     0.0     0.0
5855  0.045901  0.000000  0.035813  ...     0.0     0.0     0.0     0.0
'''
# Save the complete dataframe into csv file
complete_df.to_csv(cwd + '/Traffic data/complete_data.csv', sep=',')

# Incorporate future demand in the training data (and artificially fill
# missing demand data)
print("Pre-process training data")
tr_df.insert(tr_df.shape[1], 't_demand', value=tr_df['demand'])
tr_df.insert(tr_df.shape[1], 't_time', value=tr_df['time'].astype(int))
tr_df.insert(tr_df.shape[1], 't_geo', value=tr_df['geohash6'])

# Sort the training data based on location and time, and set the future demand
# by putting the future demand (15 minutes later) in the existing demand
# information.
tr_df.sort_values(by=['geohash6', 'time'],
                  ascending=True,
                  inplace=True)
tr_df.reset_index(drop=True, inplace=True)

# Ensure that the future demand is belong to the particular location and time,
# then proceed setting the future demand.
tr_df[['t_demand', 't_time', 't_geo']] = \
    tr_df[['t_demand', 't_time', 't_geo']].shift(-1)
# Resulting training data with future demand:
'''
        geohash6  day  timestamp    demand  ...  time  t_demand  t_time   t_geo
0         qp02yc    1         11  0.020592  ...    11  0.010292    12.0  qp02yc
1         qp02yc    1         12  0.010292  ...    12  0.006676    16.0  qp02yc
2         qp02yc    1         16  0.006676  ...    16  0.003822    18.0  qp02yc
3         qp02yc    1         18  0.003822  ...    18  0.011131    27.0  qp02yc
4         qp02yc    1         27  0.011131  ...    27  0.013487    45.0  qp02yc
5         qp02yc    1         45  0.013487  ...    45  0.003709    48.0  qp02yc
...
4206318   qp0dnn   60         44  0.045466  ...  5708  0.029285  5709.0  qp0dnn
4206319   qp0dnn   60         45  0.029285  ...  5709  0.000896  5796.0  qp0dnn
4206320   qp0dnn   61         36  0.000896  ...  5796       NaN     NaN     NaN
'''

# # Activate to ignore missing future demand data when constructing training
# # set.
# tr_df = tr_df.loc[(tr_df['time'] + 1 == tr_df['t_time']) &
#                     (tr_df['geohash6'] == tr_df['t_geo'])]
# tr_df.sort_values(by = ['geohash6', 'time'],
#                   ascending = True,
#                   inplace = True)
# tr_df.reset_index(drop = True, inplace = True)

# Activate to assume missing demand data as zero demand.
tr_df.loc[(tr_df['time'] + 1 != tr_df['t_time']) |
          (tr_df['geohash6'] != tr_df['t_geo']),
          't_demand'] = 0.0

# Remove unnecessary columns used during the incorporation of future demand
# data.
tr_df.drop(['t_time', 't_geo'], axis=1, inplace=True)

# Clean data from any NaN value.
tr_df.dropna(inplace=True)

# Save the pre-processed training data into csv file.
tr_df.to_csv(cwd + '/Traffic data/prefinal_training.csv', sep=',')
