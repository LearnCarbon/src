from flask import Flask
import ghhops_server as hs
import callingfunction

# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/GetSTRUCTUREPrediction",
    name="GetSTRUCTUREPrediction",
    description="Predicts the CO2 emission of chosen building type",
    #icon="learncarbon_logo_without_text.png",
    inputs=[
        #hs.HopsBoolean("Run prediction", "RP", "Run the prediction"),
        hs.HopsNumber("CO2", "WHAAAAAAAAAAAT", "Choose CO2 emission"),
        hs.HopsInteger("Type_B", "BT", "Input building type"),
        hs.HopsInteger("Location_B", "L", "Input choose location pf building"),
        hs.HopsNumber("Area", "A", "Input total gross area"),
        hs.HopsNumber("Floors", "FC", "Input floor count"),
    ],
    outputs=[
        hs.HopsNumber("construction_type", "P", "Structure prediction")
    ]
)


def getConstructionType(CO2, Type_B, Location_B, Area, Floors):
    prediction = callingfunction.RunPredictionOp2(CO2, Type_B, Location_B, Area, Floors)

    construction_type = prediction

    return construction_type


if __name__ == "__main__":
    app.run()