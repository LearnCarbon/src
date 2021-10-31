from flask import Flask
import ghhops_server as hs
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


# function option 1

def RunPredictionOp1(constructionType,buildingType,location,area,floorCount):

    #Load saved model
    model = tf.keras.models.load_model('LearnModel_1.h5')
    model.summary()

    # Input 1 - Building Concstruction Type
    # 0 - RC
    # 1 - Steel Concrete
    # 2 - Wood
    # 3 - Wood Hybrid
    #constructionType = 2

    # Input 2 - building type is 0 or 1
    #buildingType = 1

    # Input 3 - location 0 - 5
    #location = 3

    # Input 4 - Total floor area 
    #area = 100.0

    # Input 5 - floorCount usually between 5 and 25
    #floorCount = 10

    # Create 2d numpy array 
    inputData = np.array([[constructionType], [buildingType], [location], [area], [floorCount]])

    #pathInp = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\query.txt"
    #inputData = np.genfromtxt(pathInp)

    scalerY = joblib.load(r"C:\Users\karimd\source\repos\AEC_Hackathon2021\scalerY_1.pkl")
    scalerX = joblib.load(r"C:\Users\karimd\source\repos\AEC_Hackathon2021\scalerx_1.pkl")

    # reshaping model 
    inputData_new = np.reshape(inputData, (1,5))
    
    x_scaled = scalerX.transform(inputData_new)

    prediction = model.predict(x_scaled)
    #reshape or normalize here 
    y_scaled = scalerY.inverse_transform(prediction)

    numPrediction = float(y_scaled[0][0])
    #print(numPrediction)

    #pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
    #np.savetxt(pathOut, y_scaled)

    return numPrediction



# function option 2

def RunPredictionOp2(CO2,Type_B,Location_B,Area,Floors):

    #Load saved model
    model = tf.keras.models.load_model('LearnModel_2.h5')
    model.summary()
    
    # Input 1 - CO2 Emissions Numeical input
    #CO2 = 2

    # Input 2 - building type is 0 or 1
    #Type_B = 1

    # Input 3 - location 0 - 5
    #Location_B = 3

    # Input 4 - Total floor area 
    #Area = 100

    # Input 5 - floorCount usually between 5 and 25
    #Floors = 10

    # Create 2d numpy array 
    inputData = np.array([[CO2], [Type_B], [Location_B], [Area], [Floors]])

    #pathInp = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\query.txt"
    #inputData = np.genfromtxt(pathInp)

    #CHANGE THE FILE LOCATION!!
    scalerB = joblib.load(r"C:\Users\karimd\source\repos\AEC_Hackathon2021\scalerx_B.pkl")

    # reshaping model 
    inputData_new = np.reshape(inputData, (1,5))
    
    x_scaled = scalerB.transform(inputData_new)

    y_pred = model.predict(x_scaled)
    
    construction_type = y_pred.argmax(axis=1)
    print("this is the construction type")
    print(construction_type)
    #NOT SURE IF THE FLOAT IS NECESSARY HERE
    construction_type_unwarpped = construction_type[0]
    print(construction_type_unwarpped)

    # check which construction types each integer is
    if construction_type_unwarpped == 0:
        construction_type_string = "Concrete"

    if construction_type_unwarpped == 1:
        construction_type_string = "Steel-Concrete"
    
    if construction_type_unwarpped == 2:
        construction_type_string = "Wood"

    if construction_type_unwarpped == 3:
        construction_type_string = "Wood-Hybrid"

    print(construction_type_string)
    print(Area)
    print(Floors)

    #print(numPrediction)
    #pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
    #np.savetxt(pathOut, construction_type)

    return construction_type_string







