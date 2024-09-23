# r: tensorflow
import tensorflow as tf
# r: joblib
import joblib
# r: numpy
import numpy as np
# r: scikit-learn
import sklearn

import os
# r: argparse
import argparse

print("NumPy version:", np.__version__)

parser = argparse.ArgumentParser(description="Run machine learning model predictions")

# Adding arguments for both modelA and modelB
parser.add_argument("--targetCo2", type=float, help="Target CO2 for modelB")
parser.add_argument("--constructionType", type=int, help="Construction type (0-3) for modelA")
parser.add_argument("--buildingType", type=int, help="Building type (0 or 1) for modelA")
parser.add_argument("--location", type=int, help="Location (0-5) for modelA")
parser.add_argument("--area", type=float, help="Total floor area for modelA")
parser.add_argument("--floorCount", type=int, help="Floor count (usually between 5 and 25) for modelA")

# Parse arguments from the command line
args = parser.parse_args()

# region COnfihuring dirs
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dynamically find paths based on the project directory
log_dir = os.path.join(current_dir, 'log')
model_path = os.path.join(log_dir, 'LearnModel_A.h5')
scalerY_path = os.path.join(log_dir, 'scalerY_A.pkl')
scalerX_path = os.path.join(log_dir, 'scalerx_A.pkl')
result_file_path = os.path.join(log_dir, 'prediction_result.txt')
# endregion
#Load saved model
model = tf.keras.models.load_model(model_path)

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
    scalerX = joblib.load(scalerX_path)
    
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

if args.targetCo2 is not None:
    # If targetCo2 is provided, run modelB
    modelB(args.targetCo2)
elif all([args.constructionType is not None, args.buildingType is not None, 
          args.location is not None, args.area is not None, args.floorCount is not None]):
    # If all the required arguments for modelA are provided, run modelA
    modelA(args.constructionType, args.buildingType, args.location, args.area, args.floorCount)
else:
    print("Invalid arguments. Please provide either targetCo2 or all arguments for modelA.")
