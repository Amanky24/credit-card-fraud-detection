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


# --- Exploratory Data Analysis ---
if __name__ == '__main__':
    # Everything inside this block ONLY runs if you execute this specific file in the terminal.
    # It will NOT run when you import prepare_data into your train_model.py file.
    
    data = pd.read_csv('data/creditcard.csv')
    
    print(data.head())
    print(data.describe())
    
    # Analyzing the data for the number of fraud cases vs valid cases
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlierFraction = len(fraud)/float(len(valid))
    
    print(outlierFraction)
    print('Fraud Cases: {}'.format(len(fraud)))
    print('Valid Transactions: {}'.format(len(valid)))
    
    # Exploring the transaction amount
    print("Amount details of the fraudulent transaction")
    print(fraud.Amount.describe())
    print("Details of valid transaction")
    print(valid.Amount.describe())
    
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]
    print(X.shape)
    print(Y.shape)
    
    print("\n Data preprocessing script ran successfully!")