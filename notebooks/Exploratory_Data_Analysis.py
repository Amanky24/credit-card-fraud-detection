import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()
    
X = data.drop(['Class'], axis=1)
Y = data["Class"]
print(X.shape)
print(Y.shape)