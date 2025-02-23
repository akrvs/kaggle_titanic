import pandas as pd
import joblib

# Load the test dataset
test_df = pd.read_csv("/Users/akrvs/Projects/test.csv")

# Save PassengerId for submission
passenger_ids = test_df["PassengerId"]

# Handle missing values in 'Age' using the same approach as training
test_df['Age'] = test_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Drop columns not used in training
test_df = test_df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])
test_df['Fare'] = test_df['Fare'].fillna(0)

# One-hot encoding categorical features (ensure consistency with training data)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Load the trained model
model = joblib.load("keras_model.pkl")
print("Model loaded successfully.")

# Ensure test_df columns match model's expected features
expected_features = model.feature_names_in_  # Retrieves feature names used in training
for col in expected_features:
    if col not in test_df.columns:
        test_df[col] = 0  # Add missing columns with default value 0

# Reorder columns to match training data
test_df = test_df[expected_features]

# Make predictions
test_predictions = model.predict(test_df)
test_predictions = (test_predictions > 0.5).astype(int).flatten()

# Create a submission DataFrame
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": test_predictions})

# Save the submission file
submission.to_csv("/Users/akrvs/Projects/submission_2.csv", index=False, sep=",", encoding="utf-8")
print("Submission file saved successfully.")
