import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import altair as alt
import tensorflow as tf
import joblib
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Load clean data
data = pd.read_csv(r"Data/final_data.csv")
# The two types of data we have
data_numerical = data[["CO2", "clear_floors", "clear_Area"]]
data_categorical = data[['Type_B', 'Location_B', 'Building_Construction_Type']]
sns.set(rc = {'figure.figsize':(20,8)})
pairplot = sns.pairplot(data_numerical)

# Check the correlations between features
heatmap = sns.heatmap(data.corr())
heatmap.figure.savefig("/content/temporary/Data_correlations.png")

# Declare features
X = data.iloc[:,data.columns != 'CO2']

# We have different domains so we need to scale the data
scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)
# Target output
y = data.loc[:,"CO2"].to_numpy()
y = y.reshape(-1, 1)
# try to scale y, it was better but you need to transform it to np array and reshape it
scalerY = MinMaxScaler()


#In this case it makes sense to use MinMax scaling because the wage seems like a relative range
# Apply the scaler to our Y-features
y_scaled = scalerY.fit_transform(y)

print(y.shape)
print(y_scaled.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 21)

# Model architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50, input_shape=(n_cols,), activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
# Last dense layer with 1 output
model.add(tf.keras.layers.Dense(1, activation= "sigmoid"))
model.compile(optimizer = "adam", loss = "mean_squared_error")

model.summary()

history = model.fit(X_train,y_train,epochs=200, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss function')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("/content/temporary/ModelA_train.png")

loss_test = model.evaluate(X_test,y_test)
print('mse_test:', loss_test)

y_pred = scalerY.inverse_transform(model.predict(X_test))
y_truth = scalerY.inverse_transform(y_test)

plt.scatter(y_truth,y_pred)
plt.xlabel('Ground truth values')
_ = plt.ylabel('Predictions')

error = y_pred - y_truth
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')

# Save model and scalers
model.save("/log/Model_A/LearnModel.h5")
joblib.dump(scalerY, '/log/Model_A/scalerY.pkl')
joblib.dump(scalerX, '/log/Model_A/scalerx.pkl')
