# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Update path if needed


# Step 1:Exploration
print("----- Dataset Info -----")
print(df.info())

print("\n----- Missing Values -----")
print(df.isnull().sum())

print("\n----- First 5 Rows -----")
print(df.head())

# Step 2: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# Step 3: Encode Categorical Variables


# Label Encoding for binary column 'Sex'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Drop irrelevant columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# Step 4: Normalize/Standardize

scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])


# Step 5: Outlier Detection & Removal

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Visualize and remove outliers for each numerical column
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

    # Remove outliers
    df = remove_outliers(df, col)


# Final Output

print("\n----- Cleaned Dataset Info -----")
print(df.info())

print("\n----- Cleaned Dataset Preview -----")
print(df.head())
