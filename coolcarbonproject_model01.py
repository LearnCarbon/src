from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import altair as alt
import tensorflow as tf
import joblib

from pandas.core.indexing import check_bool_indexer

pd.set_option('display.float_format', lambda x: '%.3f' % x) # turn off scientific notation and too much decimal blah
import seaborn as sns # For pretty dataviz
sns.set_style("darkgrid") # Define style for dataviz

df = pd.read_csv(r"C:\Users\ppou\source\repos\AEC_Hackathon2021\Data\Data-Set-Material-Categories.csv")


type_local = df['Building Type'].values
location = df['Location'].values
area = df['Foot print (m2)'].values
floors = df['Floor Count'].values
buld_life = df['Life Time (Years)'].values
co2 = df['TC02 (Normalized at 50 years)'].values
material = df['Building_Construction_Type']

FData = []

for i in range(len(type_local)):
  input = type_local[i], location[i], area[i], floors[i],buld_life[i],co2[i],material[i]
  FData.append(input)

dframe = pd.DataFrame(FData)  
#print(dframe)

#Droping the empty rows
clearedDf = dframe.dropna()

clearedDf = clearedDf.rename(columns={0 : 'Type', 1: 'Location', 2 : 'Area', 3 : 'Floors', 4 : 'Build_life', 5 : 'CO2', 6 : 'Building_Construction_Type'})

clearedDf = clearedDf[clearedDf.Type != 'Generic']
clearedDf = clearedDf[clearedDf.Type != 'Industrial']

#Saving it to the csv file 
#clearedDf.to_csv('clearedData.csv',index=False)

clearedDf.info()
clearedDf.head()

clearedDf.Type = pd.Categorical(clearedDf.Type)
clearedDf['Type_B'] = clearedDf.Type.cat.codes

clearedDf.Location = pd.Categorical(clearedDf.Location)
clearedDf['Location_B'] = clearedDf.Location.cat.codes

clearedDf.Area = pd.Categorical(clearedDf.Area)
clearedDf['Area_B'] = clearedDf.Area.cat.codes

clearedDf.Floors = pd.Categorical(clearedDf.Floors)
clearedDf['Floors_B'] = clearedDf.Floors.cat.codes

clearedDf.Building_Construction_Type = pd.Categorical(clearedDf.Building_Construction_Type)
clearedDf['Building_Construction_Type'] = clearedDf.Building_Construction_Type.cat.codes


pd.set_option('display.max_columns', None)
print(clearedDf.info())
print(clearedDf.head())

Area_B = clearedDf['Area_B'].values
CO2 = clearedDf['CO2'].values

# We want to convert integers into floats based on the categories of FOOTPRINT_B, TOT_AREA_B, FLOORS_B
#                     -----0-----   -----1----  ------2----- etc.
footprint_bounds_cat = [18581, 46451, 2324, 4645, 46452, 92903, 4646, 9290, 466, 929, 9291, 18580, 0, 93, 930, 2323, 94, 465, 92903, 180000]
floor_bounds_cat = [1, 6, 15, 25, 7, 14, 25, 50]


def pick_value(column_feature, array_bounds, column_carbon):
  # print("Double of categories: ", (len(array_bounds)/2))
  # print("Number of categories: ", int(len(array_bounds)/2))
  test_array = np.empty(len(column_feature))
  for i in range(int(len(array_bounds)/2)):
    #print("category ", i)
    lower_bound = float(array_bounds[i*2])
    upper_bound = float(array_bounds[(i*2) +1])
    # print("lower bound", lower_bound)
    # print("upper bound", upper_bound)
    # create the array with carbon
    array_carbon = []
    for j in range(len(column_feature)):
      # print("problem index", j)
      # print(column_feature[j])
      if(column_feature[j] == i):
        array_carbon.append(column_carbon[j])

    
    # print("number of entries: ", len(array_carbon))
    CO_lower_bound = min(array_carbon)
    CO_upper_bound = max(array_carbon)

    # print("CO lower bound", CO_lower_bound)
    # print("CO upper bound", CO_upper_bound)

    for k in range(len(column_feature)):
      if(column_feature[k] == i):
        #print("before map", column_feature[k])
        # translate the carbon value to the range of the feature
        #column_feature[k] = float(lower_bound +((upper_bound - lower_bound) / (CO_upper_bound - CO_lower_bound))*(column_carbon[k] - CO_lower_bound))
        test_array[k] = round(float(lower_bound +((upper_bound - lower_bound) / (CO_upper_bound - CO_lower_bound))*(column_carbon[k] - CO_lower_bound)), 2)
        # print("co2 ", column_carbon[k])
        # print("after map", test_array[k])
    
  return test_array

column_c_np = np.asarray(CO2)
# AREA
column_f_np = np.asarray(Area_B)
test_area_mapped_fo = pick_value(column_f_np, footprint_bounds_cat, column_c_np)
clearedDf['clear_Area'] = test_area_mapped_fo

# FLOORS
column_floors = np.asarray(clearedDf['Floors_B'].values)
test_area_mapped_fl = pick_value(column_floors, floor_bounds_cat, column_c_np)
clearedDf['clear_floors'] = test_area_mapped_fl


pd.set_option('display.max_columns', None)
data = clearedDf
data.drop(columns=['Type', 'Location', 'Area', 'Floors',  'Build_life', 'Area_B', 'Floors_B', 'Area_B'], inplace=True)
#data = data.rename(columns={0 : 'CO2', 1: 'Building_Construction_Type', 2 : 'Type', 3 : 'Location', 4 : 'Area', 5 : 'Floors'}, inplace=True)
print(data.info())
print(data.head())
#print(data.describe())
#data.to_csv("final_data.csv", index=False)

"""NOW WE FINALLY HAVE THE CLEAN DATASET"""

data_numerical = data[["CO2", "clear_floors", "clear_Area"]]
data_categorical = data[['Type_B', 'Location_B', 'Building_Construction_Type']]

sns.pairplot(data_numerical)

sns.heatmap(data.corr())

#declare features
# from one to the last one
X = data.iloc[:,data.columns != 'CO2']
print(X)
print(X.shape)

# Load and instantiate a StandardSclaer 
from sklearn.preprocessing import StandardScaler
# different domains so we need to scale
scalerX = StandardScaler()

# Apply the scaler to our X-features
X_scaled = scalerX.fit_transform(X)

print(X_scaled.shape)

#declare regression target

y = data.loc[:,"CO2"].to_numpy()


y = y.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
# try to scale y, it was better but you need to transform it to np array and reshape it
scalerY = MinMaxScaler()


#In this case it makes sense to use MinMax scaling because the wage seems like a relative range
# Apply the scaler to our Y-features
y_scaled = scalerY.fit_transform(y)

print(y.shape)
print(y_scaled.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 21)

model = tf.keras.models.Sequential()
n_cols = X_scaled.shape[1]  

# Add 2 dense layers of 50 and 32 neurons each
model.add(tf.keras.layers.Dense(50, input_shape=(n_cols,), activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
  
# Add a dense layer with 1 value output
model.add(tf.keras.layers.Dense(1, activation= "sigmoid"))
  
# Compile your model 
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

loss_test = model.evaluate(X_test,y_test)
print('mse_test:', loss_test)

def plot_comparison(x_val, pred, truth, xlab, ylab):
  fig, ax1 = plt.subplots()
  ax1.plot(x_val, truth, color = "red", label = "truth",linestyle='None', marker = "o", markersize = 5)
  ax1.plot(x_val, pred, color = "blue", label = "pred",linestyle='None', marker = "o", markersize = 4, alpha = 0.5)

  ax1.set_xlabel(xlab)
  ax1.set_ylabel(ylab)
  ax1.legend()
  fig.set_figheight(10)
  fig.set_figwidth(20)
  plt.title('Prediction Comparison')
  plt.show()

y_pred = scalerY.inverse_transform(model.predict(X_test))
y_truth = scalerY.inverse_transform(y_test)

plt.scatter(y_truth,y_pred)

error = y_pred - y_truth
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [USD/hour]')
_ = plt.ylabel('Count')

"""SAVE MODEL AND SCALERS"""

model.save("/content/gdrive/MyDrive/temporary/LearnModel.h5")
# Save scalers
joblib.dump(scalerY, '/content/gdrive/MyDrive/temporary/scalerY.pkl')
joblib.dump(scalerX, '/content/gdrive/MyDrive/temporary/scalerx.pkl')

X_not_scaled = scalerX.inverse_transform(X)
print(X_not_scaled)