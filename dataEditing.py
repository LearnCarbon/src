from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

df= pd.read_csv('C:\\Users\\ppou\\AEC_Hackathon\\CLF Embodied Carbon Benchmark Research 17.01.31.csv')

#checking the number of empty rows in th csv file
#print (df.isnull().sum())

#Droping the empty rows
clearedDf = df.dropna()

#Saving it to the csv file 
clearedDf.to_csv('clearedData.csv',index=False)

#By area we mean footprint
area = clearedDf['$BLDG_AREA_M2'].values
#print(type(area))

#Mean value of area in m2 
area = np.where(area == '94 to 465', 279, area)
area = np.where(area == '466 to 929', 697, area)
area = np.where(area == '930 to 2323', 1626, area)
area = np.where(area == '2324 to 4645', 3484, area)
area = np.where(area == '4646 to 9290', 6968, area)
area = np.where(area == '9291 to 18580', 13935, area)
area = np.where(area == '18581 to 46451', 32021, area)
area = np.where(area == '46452 to 92903', 69677, area)
area = np.where(area == 'Over 92903', 100000, area)

#print (area)

totFloorArea = clearedDf['$BLDG_AREA_FT2'].values
#print(totFloorArea)
#print(type(totFloorArea))

totFloorArea = np.where(totFloorArea == '1,001 to 5,000', 3000, totFloorArea)
totFloorArea = np.where(totFloorArea == '5,001 to 10,000', 7500, totFloorArea)
totFloorArea = np.where(totFloorArea == '10,001 to 25,000', 17500, totFloorArea)
totFloorArea = np.where(totFloorArea == '25,001 to 50,000', 37500, totFloorArea)
totFloorArea = np.where(totFloorArea == '50,001 to 100,000', 75000, totFloorArea)
totFloorArea = np.where(totFloorArea == '100,001 to 200,000', 150000, totFloorArea)
totFloorArea = np.where(totFloorArea == '200,001 to 500,000', 350000, totFloorArea)
totFloorArea = np.where(totFloorArea == '500,001 to 1 million', 750000, totFloorArea)
totFloorArea = np.where(totFloorArea == 'Over 1 million', 1000000, totFloorArea)
#print(totFloorArea)

floors = clearedDf['$BLDG_STOR_A'].values
#print(floors)
#print(type(floors))
#print(floors.size)

floors = np.where(floors == '1 to 6', 3, floors)
floors = np.where(floors == '7 to 14', 10, floors)
floors = np.where(floors == '15 to 25', 20, floors)
floors = np.where(floors == 'More than 25', 25, floors)
#print(floors)

buildingAge = clearedDf['LCA_REFPERIOD'].values
#print(buildingAge)

co2_con_emisions = clearedDf['EC_LCAA_PERM2'].values
#print(co2_con_emisions)

co2_general_emisions = clearedDf['EC_WB_EX_OPER'].values
#print(co2_general_emisions)

FData = []

for i in range(len(area)):
    input = area[i], totFloorArea[i], floors[i], buildingAge[i], co2_con_emisions[i],co2_general_emisions[i]
    FData.append(input)


dframe = pd.DataFrame(FData)  
print(dframe)

dframe.to_csv('fresh_data.csv', index=False)


