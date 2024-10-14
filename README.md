[# LearnCarbon
Empower the architects for  analysing the relationship between structure & embodied carbon in early design stages!

Looking at all the new construction that is projected to take place between now and 2040, we see the critical role embodied carbon plays.
The opportunity to change the embodied material decreases as the project progresses, since each change has a direct impact on the project cost.

Visit https://learncarbon.me/ to learn more details on the project.

LearnCarbon is a Rhino plugin that integrates two machine learning models:
* **Model A**: inputs a conceptual massing model by just a click and gathers data on the area, total built-up, structure type and predicts the Global Warming Potential
* **Model B**: inputs area, total built-up, target GWP value and predicts the suitable structure.

## Prerequisites
Also check requirements.txt file in the src repository.
* **pandas**: required to read the library.
* **numpy**: required for the matrix calculations
* **matplotlib**: Required for the data visualization.
* **seaborn**: Statistical data visualization.
* **tensorflow**: An end-to-end open source machine learning platform.
* **scikit-learn**: Simple and efficient tools for predictive data analysis.
* **joblib**: A set of tools to provide lightweight pipelining in Python
* **ghhops_server**: Install ghhops_server
* **flask**: A Python module that lets you develop web applications easily.
* **Altair**: A declarative statistical visualization library for Python.

## Procedure

![Flow Chart](https://github.com/LearnCarbon/src/blob/main/examples/LearnCarbon.drawio.png)

### Step 01 : Cleaning and Augmenting the dataset
* CLF: Embodied Carbon Benchmark study is the initial dataset used in LearnCarbon
* Synthetic data: Data transformation (categorical to numerical) and data augmentation (add structure categorization depending on the CO2 emission, and building size.

### Step 02 : Training the ML model
* Model: Training on google collab with Tensorflow and Keras
* Validating the model: Plot learning curve

### Step 03: Linking the ML Models with Rhino

This version of the LearnCarbon Rhino Plugin no longer requires **Hops through Flask**. The models can now be executed directly via Rhino's script editor.

- The Rhino plugin retrieves parameters from the 3D model designed in Rhino and user inputs.
- The ML models process these inputs and return predictions directly into Rhino, displaying the results in the LearnCarbon interface.

---

## Setup Instructions

### Prerequisites

You’ll need a **Rhino 8** license and Python for running the scripts within Rhino.

### Step-by-Step Guide

1. **Clone the Repositories:**
   Clone the LearnCarbon Rhino Plugin and the Machine Learning backend repositories into the same directory:
   ```bash
   git clone https://github.com/LearnCarbon/Tool_LearnCarbonRhinoPlugin
   git clone https://github.com/LearnCarbon/src

2. **Install Required Libraries:** In Rhino’s run `ScriptEditor`, to run the run_ML.py script from the cloned src directory. This script will install all the required Python libraries.

4. **Run the Backend:** Once the libraries are installed, the ML models are ready to be linked with the Rhino plugin. The models will process data sent from Rhino and return GWP and structural type predictions.

5. **Build and Install the Rhino Plugin:** Follow the Rhino plugin build instructions in the [Tool_LearnCarbonRhinoPlugin repository](https://github.com/LearnCarbon/Tool_LearnCarbonRhinoPlugin).

