import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

pd.set_option('display.float_format', lambda x: '%.3f' % x) # turn off scientific notation and too much decimal blah
data = pd.read_csv(r"Data/final_data")

# Check the data
data.head()

input = data.drop(['Building_Construction_Type'], axis=1)
# Set input-output
X = data.iloc[:,:]
X = X.drop(['Building_Construction_Type'], axis=1)
Y = to_categorical(data.Building_Construction_Type)

sns.set(rc = {'figure.figsize':(15,8)})
# Check data distribution
sns.pairplot(X)

scalerX = StandardScaler()

# Scale input features with a standard scaler
X_scaled = scalerX.fit_transform(X)

# Split dataset into train and split set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 21)
# Model architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(5,), activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
# Train
history = model.fit(X_train,y_train,epochs=200, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

accuracy = model.evaluate(X_test,y_test)[1]
print('Accuracy: ', accuracy)

y_pred = model.predict(X_test)

confmatrix = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)) 
plot_confusion_matrix(confmatrix,colorbar=True,show_absolute=True,show_normed=True,hide_spines = True)

# SAVE THE MODEL and scaler
model.save("/content/gdrive/MyDrive/temporary/LearnModel_2.h5")
