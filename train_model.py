import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Step 1: Create a simple sample dataset
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'blood_pressure': [120, 130, 135, 140, 150, 160, 165, 170, 175, 180],
    'glucose_level': [85, 90, 100, 110, 120, 130, 140, 150, 160, 170],
    'cholesterol': [180, 190, 195, 200, 210, 220, 230, 240, 250, 260],
    'disease': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[['age', 'blood_pressure', 'glucose_level', 'cholesterol']]
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy*100:.2f}%")

os.makedirs("models", exist_ok=True)
with open("models/health_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved successfully as models/health_model.pkl")
