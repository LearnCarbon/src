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

pathOut = r"C:\Users\karimd\source\repos\AEC_Hackathon2021\Output\prediction.txt"
np.savetxt(pathOut, y_scaled)






#######################################
#BONUS HINT

#if you had a scaler in your training you should have had this
##Normalize data using standard scaling
#scalerY = StandardScaler().fit(outputArr)
#y_scaled = scalerY.transform(outputArr)
#print("y_scaled", np.amin(y_scaled), np.amax(y_scaled))

##Save scaler model for later use
#joblib.dump(scalerY, 'scalerY.pkl')

#SO YOU NEED TO
#Load scaler for inverse transformation
#scalerY = joblib.load("scalerY.pkl")
#And apply it to either input or output

#########################################