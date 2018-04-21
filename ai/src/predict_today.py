import math
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import  model_from_json

# Create data for today
index = pd.date_range(start='2018-04-21 00:00', end='2018-04-22 00:00',freq='15min')
data = pd.DataFrame(index=index)
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['Hour'] = data.index.hour
data['Minute'] = data.index.minute
data['Power'] = np.zeros(len(data))

# Load scaler
with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

transformed_data = scaler.transform(data)
data = pd.DataFrame(transformed_data, index=data.index, columns=data.columns)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")



prediction = np.squeeze(model.predict(data[data.columns[:-1]]), 1)

data['Power'] = prediction
data = pd.DataFrame(scaler.inverse_transform(data), index=data.index, columns=data.columns)
result = pd.DataFrame(data['Power'], index=data.index, columns=['Power'])
result.index.name = 'Date'
result.to_csv('stromverbrauch_heute.csv')