from flask import Flask
import ghhops_server as hs
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import myMl

# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/getPrediction",
    name="GetCO2Prediction",
    description="Predicts the CO2 emission of chosen building type",
    icon="learncarbon_logo_without_text.png",
    inputs=[
        hs.HopsInteger("constructionType", "ConstructionType", "Input Choose construction type"),
        hs.HopsInteger("buildingType", "BuildingType", "Input Choose building type"),
        hs.HopsInteger("location", "Location", "Input Choose location of project"),
        hs.HopsNumber("area", "Area", "Input Total gross area of project"),
        hs.HopsNumber("floorCount", "FloorCount", "Input Floor Count")
    ],
    outputs=[
        hs.HopsNumber("C02_Prediction", "C02_Prediction", "CO2 prediction")
    ]
)
def getPrediction(constructionType,buildingType,location,area,floorCount):
    # fetch Prediction data from myML
    fetchedPrediction = myMl.RunPredictionOp1(constructionType,buildingType,location,area,floorCount)
    print("prediction done!")
    print("fetched prediction data")
    return fetchedPrediction

@hops.component(
    "/getConstructionType",
    name="GetSTRUCTUREPrediction",
    description="Predicts the type of contruction given a target C02 emission",
    icon="learncarbon_logo_without_text.png",
    inputs=[
        #hs.HopsBoolean("Run prediction", "RP", "Run the prediction"),
        hs.HopsNumber("CO2", "TargetC02", "Choose target CO2 emission"),
        hs.HopsInteger("buildingType_b", "BuildingType", "Input building type"),
        hs.HopsInteger("location_b", "Location", "Input choose location pf building"),
        hs.HopsNumber("area_b", "TotalArea", "Input total gross area"),
        hs.HopsInteger("floorCount_b", "FloorCount", "Input floor count"),
    ],
    outputs=[
        hs.HopsString("construction_type", "ConstructionType_Prediction", "Construction type prediction")
    ]
)
def getConstructionType(CO2, buildingType_b, location_b, area_b, floorCount_b):
    prediction = myMl.RunPredictionOp2(CO2, buildingType_b, location_b, area_b, floorCount_b)

    construction_type = prediction

    return construction_type





if __name__ == "__main__":
    app.run()

