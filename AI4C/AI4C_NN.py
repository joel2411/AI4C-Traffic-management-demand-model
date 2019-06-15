# numpy version 1.16.3
# pandas version 0.24.2
# h5py version 2.9.0
# scikit-learn version 0.21.2
# keras version 2.2.4
import numpy as np
import h5py
import os
from sklearn.model_selection import KFold
from keras.models import Model, load_model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam

'''
This program aims to achieve the following:
- Build the demand forecasting model based on the artificial neural network
  using keras.

The input file for this program is:
- final_fullset.h5: the final training data to be used in training the 
                    forecasting model.

The output file from this program is:
- ChanceModel.h5: The resulting artificial neural network as theforecasting 
                  model built and compiled using keras.

Note :
- The input training data must be located within the folder 'Traffic data' 
  within the same directory and named as 'training.csv'. 
- This program uses keras to build, compile and train the artificial neural 
  network as the forecasting model.
- This program uses scikit-learn toolbox to enhance the training of the
  forecasting model.
- This program attempts to write a hdf5 file.
'''

# Get working directory.
cwd = os.getcwd()

# Get final training data set.
hf = h5py.File(cwd + '/Traffic data/final_fullset.h5', 'r')
fullset = hf.get('fullset')
print(fullset.shape)

# Prepare the input and target data sets.
num_train = fullset.shape[0]
quarterly_set = fullset[:num_train, [1,2,3, 4,5,6, 10,11,12]]
dc_quarterly_set = fullset[:num_train, [1,2,3, 7,8,9, 10,11,12]]
demand_set = np.reshape(fullset[:num_train, 0], (-1, 1))
pred_set = np.reshape(fullset[:num_train, 13], (-1, 1))

# Build artificial neural network (ANN) model.
# The ANN uses three input inserted at different layers.
# There are three main layers in the ANN:
# - The first main layer is used to understand the pattern of demand
#   characteristic at specific location, weekday and time.
# - The second main layer is used to understand the pattern of demand change
#   characteristic at specific location, weekday and time given the knowledge
#   extracted from the first main layer.
# - The last main layer is used to inform the current demand and make
#   relationship with the knowledge extracted from the previous two main
#   layers.
X_q_input = Input(shape=(quarterly_set.shape[1], ))
X_dc_input = Input(shape=(dc_quarterly_set.shape[1], ))
X_d_input = Input(shape=(demand_set.shape[1], ))

# First main layer.
X = Dense(64, activation='relu',
          input_dim=quarterly_set.shape[1])(X_q_input)
X = Dense(4, activation='relu')(X)
m_q = Model(inputs=X_q_input,
            outputs=X,
            name='ChanceNet_q')

# Second main layer.
q_dc = concatenate([m_q.output, X_dc_input])
Y = Dense(64, activation='relu')(q_dc)
Y = Dense(4, activation='relu')(Y)
m_dc = Model(inputs=[X_q_input, X_dc_input],
             outputs=Y,
             name='ChanceNet_dc')

# Third main layer.
q_d = concatenate([m_q.output, m_dc.output, X_d_input])
Z = Dense(32, activation='relu')(q_d)
Z = Dense(1, activation='relu')(Z)
model = Model(inputs=[X_q_input, X_dc_input, X_d_input],
              outputs=Z,
              name='ChanceNet_d')

# Show the ANN structure.
model.summary()

# Set the training parameter.
adam = Adam(lr=0.001)

# Compile the ANN model.
model.compile(optimizer=adam,
              loss='mean_absolute_error')

# # load model (used for re-training)
# model = load_model(cwd + '/Traffic data/ChanceModel.h5')
# model.summary()

# Train the ANN model with cross validation.
skf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in skf.split(demand_set):
    # The epoch is set as 5 as the error start to plateaued after second folds
    # training with 5 epochs.
    model.fit([quarterly_set[train_index],
               dc_quarterly_set[train_index],
               demand_set[train_index]],
              pred_set[train_index],
              epochs=5,
              batch_size=32,
              validation_split=0.1,
              verbose=2)

# Save the resulting ANN model.
model.save(cwd + '/Traffic data/ChanceModel.h5')

# Evaluate the resulting ANN model (to display to the user how 'good' the
# model is).
num_comp = 1000
comp_index = np.arange(quarterly_set.shape[0])[:num_comp]
model_preds = model.predict([quarterly_set[comp_index, :],
                             dc_quarterly_set[comp_index, :],
                             demand_set[comp_index, :]])
comparison = np.concatenate([np.reshape(pred_set[comp_index],
                                        (-1, 1)),
                             np.reshape(model_preds, (-1, 1))],
                            axis=1)
for c in comparison:
    print(c)
evl = model.evaluate([quarterly_set[comp_index, :],
                      dc_quarterly_set[comp_index, :],
                      demand_set[comp_index, :]],
                     pred_set[comp_index, :])
print("Loss = " + str(evl))
