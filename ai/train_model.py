import math

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Read in data and make columns from the data
data = pd.read_csv('power_data.csv', infer_datetime_format=True, index_col=0, parse_dates=True, dtype=float)
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['Hour'] = data.index.hour
data['Minute'] = data.index.minute
data['Second'] = data.index.second

# Reorder the data frame columns so that power is at the end
data = data[[*data.columns[1:], 'Power']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.values)
data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)

# Divide data into train and test set
train_percentage = 0.7
number_of_train_points = math.ceil(len(data) * train_percentage)

x_train = data[data.columns[:-1]][:number_of_train_points]
x_test = data[data.columns[:-1]][number_of_train_points:]
y_train = data['Power'][:number_of_train_points:]
y_test = data['Power'][number_of_train_points:]


# Build keras model
def baseline_model(number_of_neurons_first_layer=50, number_of_neurons_second_layer=0):
    # create model
    model = Sequential()
    model.add(Dense(number_of_neurons_first_layer, input_dim=6, kernel_initializer='glorot_normal', activation='relu'))
    if number_of_neurons_second_layer > 0:
        model.add(Dense(number_of_neurons_first_layer, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Set seed
seed = 7
np.random.seed(seed)

# evaluate model with standardized dataset
options = {'number_of_neurons_first_layer': 50, 'number_of_neurons_second_layer': 0}
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0, **options)
kfold = KFold(n_splits=min(len(x_train), 10), random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (-results.mean(), results.std()))
