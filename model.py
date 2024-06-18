import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import sklearn.tree as tree
import warnings
import pickle

# Load the dataset
path = "/Users/rishabhpatni/Desktop/Desktop/CROP PREDICTION/Crop_recommendation.csv"
dataset = pd.read_csv(path)

# Verify if the dataset is loaded correctly (optional)
print(dataset.head())

# Encode labels using LabelEncoder
label_encoder = preprocessing.LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['label'])

# Print descriptive statistics and the first few rows of the dataset
print(dataset.describe())

# Count the occurrences of each label in the 'label' column
print(dataset['label'].value_counts())

# Extract features and target variable
X = dataset.drop('label', axis=1)
y = dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Ignore the feature names warning
warnings.filterwarnings("ignore", category=UserWarning)

# Visualize the decision tree (optional)
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=label_encoder.classes_)

# Save the model and label encoder using Pickle
with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)