import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_titanic_data(file_path, visualize=False):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check the data and missing values
    print(df.head())
    print(df.info())
    print(df.isnull().sum())  # 'Embarked', 'Age', 'Cabin' columns have missing values

    # Drop column "Embarked" with only 2 missing values
    df = df.dropna(subset=['Embarked'])

    # Visualize survival based on gender if needed
    if visualize:
        sns.countplot(x='Survived', hue='Sex', data=df)
        plt.title("Survival Count Based on Gender")
        plt.show()

    # Handle missing values in 'Age' using 'Sex' and 'Pclass'
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

    # Drop unnecessary columns
    df = df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

    # Split into features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    return X, y
