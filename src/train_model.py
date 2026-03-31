from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import prepare_data
import joblib

print("Loading and preparing data...")
xTrain, xTest, yTrain, yTest = prepare_data()

print(f"Training data shape: {xTrain.shape}")
print(f"Testing data shape: {xTest.shape}")

model = RandomForestClassifier()

# Train the model using the training data
model.fit(xTrain, yTrain)

# Save the trained model to a file
print("\nSaving the trained model...")
joblib.dump(model, 'random_forest_model.joblib')

print("Model trained and saved successfully as 'random_forest_model.joblib'!")