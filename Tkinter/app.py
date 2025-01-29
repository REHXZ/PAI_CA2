import tkinter as tk
import pickle

# Load the saved model
with open("knn_model_with_preprocessing.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully!")

# Apply transformations and splitting data
X = merged_df.drop(columns=['is_fraud'])
y = merged_df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict using test data
y_pred = loaded_model.predict(X_test)
print("Predictions:", y_pred)
