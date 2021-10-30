from flask import Flask
import ghhops_server as hs
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


# function option 1

def RunPredictionOp1(area,buildingType,location,floorCount,constructionType):

    #Load saved model
    model = tf.keras.models.load_model('LearnModel_1.h5')
    model.summary()

    
    # Input 1 - Building Concstruction Type
    # 0 - RC
    # 1 - Steel Concrete
    # 2 - Wood
    # 3 - Wood Hybrid
    constructionType = 2

    # Input 2 - building type is 0 or 1
    buildingType = 1

    # Input 3 - location 0 - 5
    location = 3

    # Input 4 - Total floor area 
    area = 100

    # Input 5 - floorCount usually between 5 and 25
    floorCount = 10

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

    pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
    np.savetxt(pathOut, y_scaled)

    return numPrediction

def RunPredictionOp2():
    pass

def TogglePredictionOption():
    pass



# function option 2



