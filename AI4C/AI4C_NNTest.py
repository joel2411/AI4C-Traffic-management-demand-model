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
- Test the forecasting model to predict the future demand.

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

# Test the model using original training data.
tr_df = pd.read_csv(cwd + '/Traffic data/training.csv',
                    sep=',',
                    nrows=10**4)

# Select sample test data using permutation to sample data randomly.
num_sample = 1000
test_index = np.random.permutation(tr_df.shape[0])[:num_sample]

# Convert the data into the suitable input data a for the forecasting model.
x_input = AI4C_InputIF.elaborate_input(tr_df.values[test_index, :])
# Prepare input data sets according to forecasting model specification.
q_input = x_input[:, [1,2,3, 4,5,6, 10,11,12]]
dc_q_input = x_input[:, [1,2,3, 7,8,9, 10,11,12]]
d_input = x_input[:, 0]

# ChanceModel - NN
# Load model.
model = load_model(cwd + '/Traffic data/ChanceModel.h5')
model.summary()
# Predict the future demand.
model_preds = model.predict([q_input,
                             dc_q_input,
                             d_input])
# Show the prediction results.
print(model_preds)
