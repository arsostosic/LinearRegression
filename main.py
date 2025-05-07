import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Loading dataset

dataset = fetch_california_housing()

# Creating a dataframe to work with data from our dataset

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["PRICE"] = dataset.target # this will be our column with a value for house price .target are actually real house prices that will be inputted into our newly created column PRICE

# Checking for null values

print(df.isnull().sum())

# Splitting dataset columns X-independent, Y-dependent (PRICE - column)

x = df.drop(columns="PRICE") # means that all the other columns will be considered as independent or X values
y = df["PRICE"]

# Setting up our training and testing labels/sets

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) # this means that 0.2 or 20%
# of our whole dataset will be used for testing (X,Y values included)
# and random 42 is to enable us to have same training and testing labels every time we start code and not random again

# Scaling the values (with mean of 0 and standard deviation of 1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) # our x_train set will be scaled
x_test_scaled = scaler.fit_transform(x_test)

# Training our model using mathematical algorithm of Linear Regression

m_alg = LinearRegression()
m_alg.fit(x_train_scaled,y_train)


y_predict = m_alg.predict(x_test_scaled)


# Evaluating our model
mse = mean_squared_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

# Printing model successfulness
print(f"MSE error: {mse:.2f}")
print(f"RMSE error: {r2:.2f}")

# Prediction
# Using model to predict for random input instance

# our instance, but only one column so its just specs for one house, if we want more houses we will add more elements into the rows and more columns
sample_data = pd.DataFrame({
    "MedInc": [3.2,3.0],
    "HouseAge": [10,15],
    "AveRooms": [6,3],
    "AveBedrms": [2,1],
    "Population": [1000,1000],
    "AveOccup": [10,4],
    "Latitude": [34.05, 34.05],
    "Longitude": [-118.25,-120.75]
})

# Scaling sample data
sample_data_scaled = scaler.transform(sample_data)

# Predicting using our pretrained model m_alg

sample_prediction = m_alg.predict(sample_data_scaled)

# Printing prediction results

print(f"Predicted house price based on your data: {sample_prediction[1]*10000:.2f} $")
