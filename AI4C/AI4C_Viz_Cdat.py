# numpy version 1.16.3
# pandas version 0.24.2
# matplotlib version 3.1.0
# folium version 0.9.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import os

'''
This program aims to achieve the following:
- Extract and visualize any hidden information and pattern in the pre-processed
  training data.

The input files for this program are:
- geo_set.csv: list of unique location represented by the geohash6, latitude
               and longitude
- time_set.csv: list of unique time-key data represented by day in integer,
                timestamp in HH:mm format and time as unique time-key.
- complete_data.csv: demand data at each location and each time-key.
'''

# Get working directory.
cwd = os.getcwd()

# Load geo-location data as pandas dataframe.
print("Load geo spatial data")
geo_df = pd.read_csv(cwd + '/Traffic data/geo_set.csv',
                     sep=',')

# Load time-key list as pandas dataframe.
print("Load time list")
time_df = pd.read_csv(cwd + '/Traffic data/time_set.csv',
                      sep=',',
                      index_col=0)
time_span = time_df.shape[0]

# Load 'complete' data containing the demand information at each location
# and unique time.
print("Load complete data")
complete_df = pd.read_csv(cwd + '/Traffic data/complete_data.csv', sep=',')
h_period = 4

# Visualize the geographical locations.
print("Viz map")
locations = geo_df[['latitude', 'longitude']].values.tolist()
mymap = folium.Map(location=[-5.29, 90.7], zoom_start=12)
for point in range(0, len(locations)):
    folium.Marker(locations[point], popup=geo_df.at[point, 'geohash6']).add_to(mymap)
mymap.save(cwd + '/Traffic data/loc_map.html')

# Demand is cyclic over time.
# Visualize change of demand over time (including missing data).
print("Viz change of demand over time (including missing data)")
days = 5
_, ax = plt.subplots()
for loc in list(np.arange(5)):
    temp_df = complete_df[geo_df['geohash6'][loc]][
        time_df.loc[time_df['day'] <= days]['time']].copy()
    ax.bar(x=time_df.loc[time_df['day'] <= days]['time'],
           height=temp_df.values,
           **{'alpha': 0.5})
    ax.set_title("Change of demand over time at first 5 locs")

_, ax = plt.subplots()
for loc in list(np.random.randint(5, geo_df.shape[0], 5)):
    temp_df = complete_df[geo_df['geohash6'][loc]][
        time_df.loc[time_df['day'] <= days]['time']].copy()
    ax.bar(x=time_df.loc[time_df['day'] <= days]['time'],
           height=temp_df.values,
           **{'alpha': 0.5})
    ax.set_title("Change of demand over time at random 5 locs")

# There is no strong influence between one loc and another
# (I don't want to take a taxi because people next block are taking taxi).
# Visualize mean and sd of demands in a day.
print("Viz average and sd of demands in a day")
for g in range(5):
    _, ax = plt.subplots()
    temp_geo = geo_df.at[np.random.randint(0, geo_df.shape[0]), 'geohash6']
    temp_df = complete_df[temp_geo].copy().to_frame()
    temp_df.rename(index=str,
                   columns={temp_geo: "demand"},
                   inplace=True)
    temp_df.insert(temp_df.shape[1],
                   column='timestamp',
                   value=temp_df.index.values.astype(int))
    temp_df['timestamp'] = temp_df['timestamp'] % (24 * h_period)
    mean_temp_df = temp_df.groupby(['timestamp'])['demand'].agg(np.mean)
    std_temp_df = temp_df.groupby(['timestamp'])['demand'].agg(np.std)
    ax.bar(x=mean_temp_df.index,
           height=mean_temp_df.values,
           yerr=std_temp_df.values)
    ax.set_title("mean and sd of demand at: " + temp_geo)

# Weekday matters! (monday and sunday are different day after all).
# Visualize mean and sd of demands in a day based on weekday.
print("Viz average and sd of demands in a day based on weekday")
temp_geo = geo_df.at[np.random.randint(0, geo_df.shape[0]), 'geohash6']
temp_df = complete_df[temp_geo].copy().to_frame()
temp_df.rename(index=str,
               columns={temp_geo: "demand"},
               inplace=True)
temp_df = temp_df.join(
    pd.DataFrame(data=np.transpose([np.arange(temp_df.shape[0]),
                                    np.arange(temp_df.shape[0])]),
                 columns={'timestamp', 'day'},
                 index=np.arange(temp_df.shape[0]).astype(str)))
temp_df['timestamp'] = temp_df['timestamp'] % (24 * h_period)
temp_df['day'] = temp_df['day'] // (24 * h_period) % 7
for g in range(7):
    _, ax = plt.subplots()
    mean_temp_df = temp_df.loc[temp_df['day'] == g].\
        groupby(['timestamp'])['demand'].agg(np.mean)
    std_temp_df = temp_df.loc[temp_df['day'] == g].\
        groupby(['timestamp'])['demand'].agg(np.std)
    ax.bar(x=mean_temp_df.index,
           height=mean_temp_df.values,
           yerr=std_temp_df.values,
           alpha=0.8)
    ax.set_title("mean and sd of demand at: " +
                 temp_geo + " during weekday: " + str(g))

# From exactly one quarter to the next, there is a 'potential' pattern
# in demand changes
# (Demand 'might' go up or down on particular period but the data still
# requires adjustment to absorb the noises).
# The changes are more visible in per hourly period!
# Past time information and demand affect current demand!
# Consider time-series model such as RNN, GRU, LSTM, etc.
# Visualize up and down of demand over time in a weekday.
print("Viz demand up and down over time in a weekday")
temp_geo = geo_df.at[np.random.randint(0, geo_df.shape[0]), 'geohash6']
temp_df = complete_df[temp_geo].copy().to_frame()
temp_df.rename(index=str,
               columns={temp_geo: "demand"},
               inplace=True)
temp_df = temp_df.join(
    pd.DataFrame(data=np.transpose([np.arange(temp_df.shape[0]).astype(float),
                                    np.arange(temp_df.shape[0]),
                                    np.arange(temp_df.shape[0])]),
                 columns={'delta_d', 'timestamp', 'day'},
                 index=np.arange(temp_df.shape[0]).astype(str)))
temp_df['delta_d'][0] = 0.0
for i in range(temp_df.shape[0] - 1):
    temp_df['delta_d'][i + 1] = temp_df['demand'][i + 1] - temp_df['demand'][i]
temp_df['timestamp'] = temp_df['timestamp'] % (24 * h_period)
temp_df['day'] = temp_df['day'] // (24 * h_period) % 7
for g in range(7):
    _, ax = plt.subplots()
    mean_temp_df = temp_df.loc[temp_df['day'] == g].\
        groupby(['timestamp'])['delta_d'].agg(np.mean)
    std_temp_df = temp_df.loc[temp_df['day'] == g].\
        groupby(['timestamp'])['delta_d'].agg(np.std)
    ax.bar(x=mean_temp_df.index,
           height=mean_temp_df.values,
           yerr=std_temp_df.values,
           alpha=0.8)
    ax.set_title("Ups and downs of demand at: " + temp_geo +
                 " during weekday: " + str(g))


# Although there is seems to be some pattern of demand
# distribution in specific time in a weekday,
# there is not enough data to justify this pattern.
# Visualize distribution of demand on a day.
ori_temp_df = complete_df.copy()
ori_temp_df = ori_temp_df.join(
    pd.DataFrame(data=np.transpose([np.arange(ori_temp_df.shape[0]),
                                    np.arange(ori_temp_df.shape[0])]),
                 columns={'timestamp', 'day'},
                 index=np.arange(ori_temp_df.shape[0]).astype(int)))
ori_temp_df['timestamp'] = ori_temp_df['timestamp'] % (24 * h_period)
ori_temp_df['day'] = ori_temp_df['day'] // (24 * h_period) % 7

temp_geo = geo_df['geohash6'][210]
temp_df = ori_temp_df[[temp_geo, 'day', 'timestamp']].copy()
day = 0
for ts in list([32, 33, 34, 35, 36, 37, 38, 39]):
    _, ax = plt.subplots()
    day_temp_df = temp_df.loc[(temp_df['day'] == day) &
                              (temp_df[temp_geo] > 0) &
                              (temp_df['timestamp'] == ts)]
    ax.hist([day_temp_df[temp_geo].values],
            bins=10,
            alpha=0.7)
    ax.set_title("Histogram of demand at: " + temp_geo + " during: " + str(ts))

plt.show()
