from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from pandas.core.indexing import check_bool_indexer


df= pd.read_csv(r"C:\Users\ppou\source\repos\AEC_Hackathon2021\Data\raw_data.csv")

type = df['BLDG_TYP'].values
#type = np.where(type == 'Commercial', 0, type)
#type = np.where(type == 'Residential', 1, type)
location = df['BLDG_LOC_REGION'].values
footprint = df['$BLDG_AREA_M2'].values
tot_area = df['$BLDG_AREA_FT2'].values
floors = df['$BLDG_STOR_A'].values
buld_life = df['LCA_REFPERIOD'].values
co2 = df['EC_LCAA_PERM2'].values

FData = []

for i in range(len(type)):
    input = type[i], location[i], footprint[i], tot_area[i], floors[i],buld_life[i],co2[i]
    FData.append(input)


dframe = pd.DataFrame(FData)  
#print(dframe)

#Droping the empty rows
clearedDf = dframe.dropna()

clearedDf = clearedDf.rename(columns={0 : 'Type', 1: 'Location', 2 : 'Footprint', 3 : 'Tot_area', 4 : ' Floors', 5 : 'Build_life', 6 : 'CO2'})

#Saving it to the csv file 
#clearedDf.to_csv('clearedData.csv',index=False)

clearedDf.Type = pd.Categorical(clearedDf.Type)
clearedDf['Type_B'] = clearedDf.Type.cat.codes

clearedDf.Location = pd.Categorical(clearedDf.Location)
clearedDf['Location_B'] = clearedDf.Location.cat.codes

clearedDf.Footprint = pd.Categorical(clearedDf.Footprint)
clearedDf['Footprint_B'] = clearedDf.Footprint.cat.codes

clearedDf.Tot_area = pd.Categorical(clearedDf.Tot_area)
clearedDf['Tot_area_B'] = clearedDf.Tot_area.cat.codes

#print(clearedDf.info())
#print(clearedDf.head())

Footprint_B = clearedDf['Footprint_B'].values
Tot_area_B = clearedDf['Tot_area_B'].values
CO2 = clearedDf['CO2'].values

check_df = []
for i in range(len(Footprint_B)):
    if Footprint_B[i] == 3:
        input = Footprint_B[i],CO2[i]
        check_df.append(input)

print(min(check_df))
print(max(check_df))








#print(check_df)

#print(Footprint_B)
#print(Tot_area_B)
#print(CO2)





