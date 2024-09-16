# r: tensorflow
import tensorflow as tf
# r: joblib
import joblib
# r: numpy
import numpy as np
# r: scikit-learn
import sklearn

#Load saved model
model = tf.keras.models.load_model("C:/Reope/GitHub/src/log/LearnModel_A.h5")
print(model.summary())

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
area = 100.0

# Input 5 - floorCount usually between 5 and 25
floorCount = 10

inputData = np.array([[constructionType], [buildingType], [location], [area], [floorCount]])

scalerY = joblib.load('C:/Reope/GitHub/src/log/scalerY_A.pkl')
scalerX = joblib.load('C:/Reope/GitHub/src/log/scalerx_A.pkl')

# reshaping model 
inputData_new = np.reshape(inputData, (1,5))

x_scaled = scalerX.transform(inputData_new)

prediction = model.predict(x_scaled)
#reshape or normalize here 
y_scaled = scalerY.inverse_transform(prediction)

numPrediction = float(y_scaled[0][0])
print("Building 1: " + str(numPrediction))

with open('C:/Reope/GitHub/src/log/prediction_result.txt', 'w') as f:
    f.write(str(numPrediction))