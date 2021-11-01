from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import altair as alt
import tensorflow as tf
import joblib
from pandas.core.indexing import check_bool_indexer
import seaborn as sns

# Read dataset csv with the material column
df = pd.read_csv(r"Data-Set-Material-Categories.csv")

# Function to convert binary columns based on their carbon emission
def pick_value(column_feature, array_bounds, column_carbon):
  test_array = np.empty(len(column_feature))
  for i in range(int(len(array_bounds)/2)):
    lower_bound = float(array_bounds[i*2])
    upper_bound = float(array_bounds[(i*2) +1])
    array_carbon = []
    for j in range(len(column_feature)):
      if(column_feature[j] == i):
        array_carbon.append(column_carbon[j])
    CO_lower_bound = min(array_carbon)
    CO_upper_bound = max(array_carbon)

    for k in range(len(column_feature)):
      if(column_feature[k] == i):
        test_array[k] = round(float(lower_bound +((upper_bound - lower_bound) / (CO_upper_bound - CO_lower_bound))*(column_carbon[k] - CO_lower_bound)), 2)
  return test_array
  
#--------------- Organizing data ---------------
data = df[['Building Type', 'Location', 'Foot print (m2)', 'Floor Count', 'Life Time (Years)', 'TC02 (Normalized at 50 years)', 'Building_Construction_Type']]
# Deleting rows with no values
data = dframe.dropna()
data = data.rename(columns={0 : 'Type', 1: 'Location', 2 : 'Area', 3 : 'Floors', 4 : 'Build_life', 5 : 'CO2', 6 : 'Building_Construction_Type'})
data.head("Clear data: ", clearedDf.head())
# Convert from numerical to categorical
data.Type = pd.Categorical(clearedDf.Type)
data['Type_B'] = data.Type.cat.codes
data.Location = pd.Categorical(clearedDf.Location)
data['Location_B'] = data.Location.cat.codes
data.Area = pd.Categorical(clearedDf.Area)
data['Area_B'] = data.Area.cat.codes
data.Floors = pd.Categorical(clearedDf.Floors)
data['Floors_B'] = data.Floors.cat.codes
data.Building_Construction_Type = pd.Categorical(clearedDf.Building_Construction_Type)
data['Building_Construction_Type'] = data.Building_Construction_Type.cat.codes

# --------------- Convert the categorical columns (1 to 6 floors) to numerical based on the carbon footprint of the sample ---------------
# --------------- Indices used for the categories when pandas categorical function ---------------
footprint_bounds_cat = [18581, 46451, 2324, 4645, 46452, 92903, 4646, 9290, 466, 929, 9291, 18580, 0, 93, 930, 2323, 94, 465, 92903, 180000]
floor_bounds_cat = [1, 6, 15, 25, 7, 14, 25, 50]

# --------------- AREA ---------------
column_c_np = np.asarray(CO2)
column_f_np = np.asarray(Footprint_B)
test_area_mapped_fo = pick_value(column_f_np, footprint_bounds_cat, column_c_np)
data['clear_Area'] = test_area_mapped

# --------------- FLOORS ---------------
column_floors = np.asarray(data['Floors_B'].values)
test_area_mapped_fl = pick_value(column_floors, floor_bounds_cat, column_c_np)
data['clear_floors'] = test_area_mapped_fl

# --------------- Clean one more time ---------------
data.drop(columns=['Type', 'Location', 'Area', 'Floors',  'Build_life', 'Area_B', 'Floors_B', 'Area_B'], inplace=True)
# --------------- Save if needed ---------------
# data.to_csv("/content/gdrive/MyDrive/temporary/final_data", index=False)
