import tensorflow as tf
import numpy as np
import joblib


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
    scalerB = joblib.load(r"C:\Users\ppou\source\repos\AEC_Hackathon2021\ConnectHopsModel2\scalerx_B.pkl")

    # reshaping model 
    inputData_new = np.reshape(inputData, (1,5))
    
    x_scaled = scalerB.transform(inputData_new)

    y_pred = model.predict(x_scaled)
    
    construction_type = y_pred.argmax(axis=1)

    #NOT SURE IF THE FLOAT IS NECESSARY HERE
    construction_type = float(construction_type[0][0])


    #print(numPrediction)
    #pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
    #np.savetxt(pathOut, construction_type)

    return construction_type
