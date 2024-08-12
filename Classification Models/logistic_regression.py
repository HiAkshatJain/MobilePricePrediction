# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset into a DataFrame
data = pd.read_csv('Dataset/Mobilephone_data_final.csv')

# Remove the 'Sr. No.' column
data.drop(columns='Sr. No.', inplace=True)

# Bin the 'Price' into categories
# Define your bins and labels (you can adjust the bins as per your dataset)
bins = [0, 10000, 20000, 30000, 40000, np.inf]  # Example bins
labels = ['0-10k', '10k-20k', '20k-30k', '30k-40k', '40k+']  # Labels for each bin
data['PriceCategory'] = pd.cut(data['Price'], bins=bins, labels=labels, right=False)

# Set up feature matrix (X) and target vector (y)
x = data.copy()
y = data['PriceCategory']
x.drop(columns=['Price', 'Name', 'PriceCategory'], inplace=True)

# Encode categorical features
le = LabelEncoder()
for column in ['Brand', 'Model', 'Operating system']:
    x[column] = le.fit_transform(x[column])

# Apply OneHotEncoding to specified columns
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [4, 13, 14, 15, 17, 18])], 
    remainder='passthrough'
)

x = np.array(ct.fit_transform(x))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x,  # Feature matrix
    y,  # Target vector
    test_size=0.2,  # Proportion of the data to include in the test split (20%)
    random_state=441  # Seed for random number generation to ensure reproducibility
)

# Feature Scaling   
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the target vector
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
