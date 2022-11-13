# The objective of regression is to predict continuous values such as sales amount, quantity, temperature,
# number of customers, etc. All modules in PyCaret provide many pre-processing features to prepare the data for
# modeling through the setup function. It has over 25 ready-to-use algorithms and several plots to analyze the
# performance of trained models.


import pandas as pd

# Importing the dataset
data = pd.read_csv('AirPassengers.csv')
data['Date'] = pd.to_datetime(data['Date'])

print(data.head())
print(data.info())
print(data.describe())

# Create 12-month moving average
data['MA12'] = data['Passengers'].rolling(12).mean()

# plot the data and MA
# fig = px.line(data, x="Date", y=["Passengers", "MA12"], template='plotly_dark')
# fig.show()

# Extract Month and year from Date column
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
print(data['Month'].head())
print(data['Year'].head())

# create a sequence of numbers
data['Series'] = range(1, len(data) + 1)

# drop unnecessary columns and re-arrange
data.drop(['Date', 'MA12'], axis=1, inplace=True)
data = data[['Series', 'Year', 'Month', 'Passengers']]
print(data.head())

# Split the data into training and testing sets


# split data into train-test set
train = data[data['Year'] < 1960]
test = data[data['Year'] >= 1960]

# check shape
print(train.shape, test.shape)

# import the regression module
from pycaret.regression import *

s = setup(data=train, test_data=test, target='Passengers', fold_strategy='timeseries',
          numeric_features=['Year', 'Series'], fold=3, transform_target=True, session_id=123)
print(s)

# Train and Evaluate Model
best = compare_models(sort='MAE')
print(best)

prediction_holdout = predict_model(best)
print(prediction_holdout)

"""
X_train, X_test, y_train, y_test = train_test_split(data[['Year', 'Month']], data['Passengers'], test_size=0.2,
                                                    random_state=0)

print(X_train.shape)
print(X_test.shape)
"""
