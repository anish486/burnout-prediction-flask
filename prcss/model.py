import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

X = data.drop("burnout", axis=1)
y = data["burnout"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("burnout_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")