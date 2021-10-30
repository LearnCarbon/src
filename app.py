from flask import Flask
import ghhops_server as hs
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


#Load saved model
model = tf.keras.models.load_model('LearnModel_1.h5')
model.summary()

pathInp = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\query.txt"
inputData = np.genfromtxt(pathInp)

scalerY = joblib.load(r"C:\Users\karimd\source\repos\AEC_Hackathon2021\scalerY_1.pkl")
scalerX = joblib.load(r"C:\Users\karimd\source\repos\AEC_Hackathon2021\scalerx_1.pkl")


inputData_new = np.reshape(inputData, (1,5))
print(inputData_new)


x_scaled = scalerX.transform(inputData_new)

prediction = model.predict(x_scaled)
#reshape or normalize here 
y_scaled = scalerY.inverse_transform(prediction)

numPrediction = float(y_scaled[0][0])
print(numPrediction)


pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
np.savetxt(pathOut, y_scaled)



# this is where the hops app is

# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/getPrediction",
    name="GetCO2Prediction",
    description="Predicts the CO2 emission of chosen building type",
    #icon="learncarbon_logo_without_text.png",
    inputs=[
        hs.HopsBoolean("Run", "R", "Toggle to run prediction"),
        #hs.HopsNumber("t", "t", "Parameter on Curve to evaluate"),
    ],
    outputs=[
        hs.HopsNumber("Prediction", "P", "CO2 prediction")
    ]
)
def getPrediction(run):
    if run:
        return numPrediction
    else:
        return 0

if __name__ == "__main__":
    app.run()

