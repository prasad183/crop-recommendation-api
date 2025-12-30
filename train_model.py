import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\bhyri\Desktop\Crop_recommendation.csv')  # Update path if needed

# Features and label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train model (Random Forest is very accurate on this dataset ~99%)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, r'C:\Users\bhyri\Desktop\crop_recommendation_api\crop_model.pkl')
print("Model trained and saved successfully!")