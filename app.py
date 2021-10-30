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
    #icon="learncarbon_logo_without_text.png",
    inputs=[
        hs.HopsInteger("constructionType", "CT", "Input Choose construction type"),
        hs.HopsInteger("buildingType", "BT", "Input Choose building type"),
        hs.HopsInteger("location", "L", "Input Choose location of project"),
        hs.HopsNumber("area", "A", "Input Total gross area of project"),
        hs.HopsNumber("floorCount", "FC", "Input Floor Count")
    ],
    outputs=[
        hs.HopsNumber("Prediction", "P", "CO2 prediction")
    ]
)


def getPrediction(constructionType,buildingType,location,area,floorCount):
    # fetch Prediction data from myML
    fetchedPrediction = myMl.RunPredictionOp1(constructionType,buildingType,location,area,floorCount)
    print("fetched prediction data")
    return fetchedPrediction


if __name__ == "__main__":
    app.run()

