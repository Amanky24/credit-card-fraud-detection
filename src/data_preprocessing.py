import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split

def prepare_data(filepath='data/creditcard.csv'):
    """Loads, prepares, and splits the dataset."""
    data = pd.read_csv(filepath)
    
    # Preparing the data for the model
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]
    
    xData = X.values
    yData = Y.values
    
    # Splitting the data
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
    
    # Return the datasets so other files can grab them
    return xTrain, xTest, yTrain, yTest

