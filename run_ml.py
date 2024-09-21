# r: tensorflow
import tensorflow as tf
# r: joblib
import joblib
# r: numpy
import numpy as np
# r: scikit-learn
import sklearn
# r: sys
import sys
# r: os
import os
# r: pathlib
from pathlib import Path


# region COnfihuring dirs
current_dir = Path(__file__).resolve().parent

# Dynamically find paths based on the project directory
log_dir = current_dir / 'log'
model_path = log_dir / 'LearnModel_A.h5'
scalerY_path = log_dir / 'scalerY_A.pkl'
scalerX_path = log_dir / 'scalerx_A.pkl'
result_file_path = log_dir / 'prediction_result.txt'
# endregion
#Load saved model
model = tf.keras.models.load_model(model_path)
print(model.summary())

def modelB(targetCo2):
    print(f"Processing with targetCo2: {targetCo2}")

    with open(result_file_path, 'w') as f:
        f.write(str(targetCo2))

# Function 2: Takes all arguments
def modelA(constructionType, buildingType, location, area, floorCount):
    # Prepare input data
    inputData = np.array([[constructionType], [buildingType], [location], [area], [floorCount]])
    
    # Load scalers
    scalerY = joblib.load(scalerY_path)
    scalerX = joblib.load(scalerX_path')
    
    # Reshaping input data
    inputData_new = np.reshape(inputData, (1, 5))
    x_scaled = scalerX.transform(inputData_new)

    # Make prediction
    prediction = model.predict(x_scaled)
    
    y_scaled = scalerY.inverse_transform(prediction)
    numPrediction = float(y_scaled[0][0])
    
    print("Building 1: " + str(numPrediction))
    
    # Write the result to file
    with open(result_file_path, 'w') as f:
        f.write(str(numPrediction))

# Check how many arguments were passed from C# to identify the model
if len(sys.argv) == 2:
    targetCo2 = float(sys.argv[1])
    modelB(targetCo2)
else:
    constructionType = int(sys.argv[1])
    buildingType = int(sys.argv[2])
    location = int(sys.argv[3])
    area = float(sys.argv[4])
    floorCount = int(sys.argv[5])
    
    modelA(constructionType, buildingType, location, area, floorCount)
