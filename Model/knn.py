# Importing essential libraries for data analysis and visualization

import numpy as np  # Fundamental package for scientific computing in Python
import pandas as pd  # Powerful data manipulation and analysis library
from sklearn.compose import ColumnTransformer  # For applying transformers to specific columns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # For encoding categorical features
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and test sets
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors regression model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For evaluating the model

# Load the dataset into a DataFrame
data = pd.read_csv('Dataset/Mobilephone_data_final.csv')

# Remove the 'Sr. No.' column from the DataFrame
data.drop(columns='Sr. No.', inplace=True)

# Selection Column for model
x = data
y = data['Price']
x.drop(columns=['Price', 'Name'], inplace=True)

# Data Encoding
le = LabelEncoder()  # Initialize the LabelEncoder

# Loop through specified columns to encode categorical features
for column in ['Brand', 'Model', 'Operating system']:
    le = LabelEncoder()  # Re-initialize LabelEncoder for each column
    x[column] = le.fit_transform(x[column])  # Encode categorical column and replace in DataFrame

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [4, 13, 14, 15, 17, 18])],  # Apply OneHotEncoder to specified columns
    remainder='passthrough'  # Leave other columns unchanged
)

x = np.array(ct.fit_transform(x))  # Apply transformations and convert the result to a NumPy array

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x,  # Feature matrix
    y,  # Target vector
    test_size=0.2,  # Proportion of the data to include in the test split (20%)
    random_state=2  # Seed for random number generation to ensure reproducibility
)

# Feature Scaling   
sc = StandardScaler()  # Initialize the StandardScaler
X_train = sc.fit_transform(X_train)  # Fit to the training data and transform it
X_test = sc.transform(X_test)  # Transform the test data using the same scaler

# Initialize the KNeighborsRegressor model
knn = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed
knn.fit(X_train, y_train)  # Fit the model to the training data

# Predict the target vector 
y_pred = knn.predict(X_test)

# Evaluate the model
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate the Root Mean Squared Error (RMSE)
MAE = mean_absolute_error(y_test, y_pred)  # Calculate the Mean Absolute Error (MAE)
R_squared = r2_score(y_test, y_pred)  # Calculate the R-squared (R²) score

print(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")
print(f"Mean Absolute Error (MAE): {MAE:.4f}")  
print(f"R-squared (R²): {R_squared:.4f}")  
