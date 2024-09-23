# r: tensorflow
import tensorflow as tf
# r: joblib
import joblib
# r: numpy
import numpy as np
# r: scikit-learn
import sklearn
import os
import Rhino
# r: dill

print("NumPy version:", np.__version__)

# region COnfihuring dirs
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dynamically find paths based on the project directory
log_dir = os.path.join(current_dir, 'log')
modelA_path = os.path.join(log_dir, 'LearnModel_A.h5')
modelB_path = os.path.join(log_dir, 'LearnModel_B.h5')
scalerY_path = os.path.join(log_dir, 'scalerY_A.pkl')
scalerX_path = os.path.join(log_dir, 'scalerx_A.pkl')
scalerX_B_path = os.path.join(log_dir, 'scalerx_B.pkl')
result_file_path = os.path.join(log_dir, 'prediction_result.txt')
input_file_path = os.path.join(log_dir, 'input.txt')
# endregion

def modelB(targetCo2, buildingType, location, area, floorCount):
    Rhino.RhinoApp.WriteLine(f"Processing with targetCo2: {targetCo2}")
    model = tf.keras.models.load_model(modelB_path)
    inputData = np.array([[targetCo2], [buildingType], [location], [area], [floorCount]])

    scalerX = joblib.load(scalerX_B_path)

    inputData_new = np.reshape(inputData, (1, 5))
    x_scaled = scalerX.transform(inputData_new)

    # Make prediction
    prediction = model.predict(x_scaled)

    construction_type = prediction.argmax(axis=1)
    print("this is the construction type")
    print(construction_type)
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

    with open(result_file_path, 'w') as f:
        f.write(construction_type_string)

# Function 2: Takes all arguments
def modelA(constructionType, buildingType, location, area, floorCount):
    model = tf.keras.models.load_model(modelA_path)
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
        
    # Write the result to file
    with open(result_file_path, 'w') as f:
        f.write(str(numPrediction))
    
# Read inputs from the input file
with open(input_file_path, 'r') as f:
    inputs = f.read().strip().split(',')

if inputs[0] == "False":
    targetCo2, buildingType, location, area, floorCount = map(float, inputs[1:])
    modelB(targetCo2, buildingType, location, area, floorCount)
else:
    print(inputs)
    constructionType, buildingType, location, area, floorCount = map(float, inputs[1:])
    modelA(int(constructionType), int(buildingType), int(location), area, int(floorCount))
