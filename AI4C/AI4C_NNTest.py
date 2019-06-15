# numpy version 1.16.3
# pandas version 0.24.2
# keras version 2.2.4
import numpy as np
import pandas as pd
import os
from keras.models import load_model

import AI4C_InputIF

'''
This program aims to achieve the following:
- Test the forecasting model in predicting the future demand.

The input files for this program are:
- training.csv: training data containing geohash6, demand, day and timestamp 
                data.
- ChanceModel.h5: The resulting artificial neural network as theforecasting 
                  model built and compiled using keras.
                  
The input file for AI4C_InputIF used in this program are:
- quarterly_profile.csv: the profile data for each location as the mean and
                         standard deviation of the demand, and the number of 
                         non-zero demand at each weekday and time.
- dchange_quarterly_profile.csv: the profile data for each location as the 
                                 mean and standard deviation of the demand 
                                 at each weekday and time.
- q_pop_max.csv: contain the value used to normalize the number of non-zero 
                 demand features.

The output file from this program is:
- None

Note :
- The input training data must be located within the folder 'Traffic data' 
  within the same directory and named as 'training.csv'. 
- This program uses keras to build the artificial neural network as the 
  forecasting model.
'''

# Get working directory.
cwd = os.getcwd()

# Initialize the absolute time to be used for converting
# time data to integer.
abs_time = pd.to_datetime('00:00', format='%H:%M')

# Test the model using original training data.
tr_df = pd.read_csv(cwd + '/Traffic data/training.csv',
                    sep=',',
                    nrows=10**4)

# Select sample test data using permutation to sample data randomly.
num_sample = 1000
test_index = np.random.permutation(tr_df.shape[0])[:num_sample]

# Convert the data into the suitable input data a for the forecasting model.
x_input = AI4C_InputIF.elaborate_input(tr_df.values[test_index, :],
                                       raw=True)
# Store time and day information for T > 1 predictions.
x_ts = ((pd.to_datetime(tr_df.loc[test_index, 'timestamp'], format='%H:%M') -
         abs_time) // 15).astype('timedelta64[m]').astype(int).values
x_day = tr_df.loc[test_index, 'day'].values
# Prepare input data sets according to forecasting model specification.
q_input = x_input[:, [1, 2, 3, 4, 5, 6, 10, 11, 12]]
dc_q_input = x_input[:, [1, 2, 3, 7, 8, 9, 10, 11, 12]]
d_input = x_input[:, 0]

# Set the range of demand prediction, T + T_length.
T_length = 5

# ChanceModel - NN
# Load model.
model = load_model(cwd + '/Traffic data/ChanceModel.h5')
model.summary()
# Predict the future demand at T + 1.
model_preds = model.predict([q_input,
                             dc_q_input,
                             d_input])
# Store the prediction results.
complete_preds = model_preds

# Predict the future demand larger than the next 15 minutes
for t in range(T_length - 1):
    # update time information.
    x_ts += 1
    x_day += (x_ts // 96)
    x_ts = x_ts % 96
    # Reconstruct data using demand obtained from the previous prediction
    # and the updated time.
    x_newraw = np.concatenate((np.reshape(tr_df.values[test_index, 0],
                                          (-1, 1)),
                               np.reshape(x_day, (-1, 1)),
                               np.reshape(x_ts, (-1, 1)),
                               model_preds),
                              axis=1)
    # Convert the data into the suitable input data a for the
    # forecasting model.
    x_input = AI4C_InputIF.elaborate_input(x_newraw, raw=False)
    # Prepare input data sets according to forecasting model specification.
    q_input = x_input[:, [1, 2, 3, 4, 5, 6, 10, 11, 12]]
    dc_q_input = x_input[:, [1, 2, 3, 7, 8, 9, 10, 11, 12]]
    d_input = x_input[:, 0]
    # Predict the future demand.
    model_preds = model.predict([q_input,
                                 dc_q_input,
                                 d_input])
    # Store the prediction results.
    complete_preds = np.concatenate((complete_preds, model_preds), axis=1)

# Show prediction results.
# columns represent T + column index.
# rows represent a prediction at a location and time.
for pred in complete_preds:
    print(pred)
