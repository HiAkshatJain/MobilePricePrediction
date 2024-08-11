# Importing essential libraries for data analysis and visualization

# NumPy is a fundamental package for scientific computing in Python.
# It provides support for large, multi-dimensional arrays and matrices,
# along with a collection of mathematical functions to operate on these arrays.
import numpy as np

# pandas is a powerful data manipulation and analysis library.
# It offers data structures such as DataFrame and Series for handling and analyzing structured data.
# DataFrames are particularly useful for data cleaning, transformation, and analysis.
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset into a DataFrame
data = pd.read_csv('Dataset/Mobilephone_data_final.csv')

# Remove the 'Sr. No.' column from the DataFrame
data.drop(columns='Sr. No.', inplace=True)

# Selection Column for model
x = data
y = data['Price']
x.drop(columns=['Price','Name'], inplace=True)

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
    random_state=441  # Seed for random number generation to ensure reproducibility
)

# Feature Scaling   
sc = StandardScaler()  # Initialize the StandardScaler

X_train = sc.fit_transform(X_train)  # Fit to the training data and transform it
X_test = sc.transform(X_test)  # Transform the test data using the same scaler

# Initialize the LinearRegression model
reg = LinearRegression()
reg.fit(X_train, y_train)  # Fit the model to the training data

# Predict the target vector 
y_pred = reg.predict(X_test)

# Evaluate the model
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate the Root Mean Squared Error (RMSE)
MAE = mean_absolute_error(y_test, y_pred)  # Calculate the Mean Absolute Error (MAE)
R_squared = r2_score(y_test, y_pred)  # Calculate the R-squared (R²) score

print(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")
print(f"Mean Absolute Error (MAE): {MAE:.4f}")  
print(f"R-squared (R²): {R_squared:.4f}")  