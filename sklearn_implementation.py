import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_preprocessing import process_titanic_data

X, y = process_titanic_data("/Users/akrvs/Projects/train.csv")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make a prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy of a basic Logistic Regression model: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved successfully.")


