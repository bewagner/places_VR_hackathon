import math
import pickle

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Read in data and make columns from the data
data = pd.read_csv('electricity_data.csv', infer_datetime_format=True, index_col=0, parse_dates=True, header=None,
                   dtype=float)
data.columns = ['Power']
data.index.name = 'Date'
data.dropna(inplace=True)
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['Hour'] = data.index.hour
data['Minute'] = data.index.minute

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
    model.add(Dense(number_of_neurons_first_layer, input_dim=5, kernel_initializer='glorot_normal', activation='relu'))
    if number_of_neurons_second_layer > 0:
        model.add(Dense(number_of_neurons_first_layer, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Set seed
seed = 7
np.random.seed(seed)

train = True
old_error = 1000
best_first_layer = 0
best_second_layer = 0
best_batch_size = 0

values = [2, 13, 24, 35, 46, 56, 67, 78, 89, 100]

parameters = {'number_of_neurons_first_layer': values, 'number_of_neurons_second_layer': values,
              'batch_size': list(map(int, map(round, np.linspace(20, 300, 10))))}
estimator = KerasRegressor(build_fn=baseline_model, epochs=300, verbose=2)
clf = GridSearchCV(estimator, parameters)
clf.fit(x_train, y_train)

with open('clf.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save model
#  serialize model to JSON
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save_weights("model.h5")

"""
if train:
    for first_layer in values:
        for second_layer in values:
            for batch_size in map(round, np.linspace(20, 300, 10)):
                print('First layer', best_first_layer)
                print('Second layer', best_second_layer)
                print('Batch size', best_batch_size)

                # evaluate model with standardized dataset
                options = {'number_of_neurons_first_layer': first_layer, 'number_of_neurons_second_layer': second_layer}

                kfold = KFold(n_splits=min(len(x_train), 2), random_state=seed)

                results = cross_val_score(estimator, x_train, y_train, cv=kfold)
                print("Results: %.2f (%.2f) MSE" % (-results.mean(), results.std()))

                errors[i, j, k] = -results.mean()

                if -results.mean() < old_error:
                    old_error = -results.mean()
                    best_batch_size = batch_size
                    best_first_layer=first_layer
                    best_second_layer=second_layer

                    #estimator.fit(x_test,y_test)

                    # Save model
                    # serialize model to JSON
                    model_json = estimator.model.to_json()
                    with open("model.json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    estimator.model.save_weights("model.h5")
                k = k + 1
            j = j + 1
        i = i + 1

else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    estimator = loaded_model

np.save('errors.npy',errors)

params = np.asarray([best_first_layer, best_second_layer, best_batch_size])
np.save('hyperparams.npy', params)

print('Best error', old_error)
print('Best first layer', best_first_layer)
print('Best second layer', best_second_layer)
print('Best batch size', best_batch_size)

prediction = estimator.predict(x_test)


def inverse_data(data, scaler):
    x_test_copy = x_test.copy()

    x_test_copy['Power'] = data
    return scaler.inverse_transform(x_test_copy)[:, 5]

inversed_prediction = inverse_data(np.squeeze(prediction, 1), scaler)
inversed_y_test = inverse_data(y_test, scaler)

print(inversed_y_test.shape)
print(inversed_prediction.shape)


print(inversed_y_test[:5])
print(inversed_prediction[:5])
print(mean_squared_error(inversed_y_test, inversed_prediction))

"""
