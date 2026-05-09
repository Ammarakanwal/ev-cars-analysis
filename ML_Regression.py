import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv(r'D:\iec\portfolio2\66634-ev-data\Electric_Vehicle_Population_Data.csv')

print(df.head())

# Keep only useful columns
df = df[[
    "Model Year",
    "Make",
    "Electric Vehicle Type",
    "Base MSRP",
    "Electric Range"
]]

# Drop missing values
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, columns=["Make", "Electric Vehicle Type"], drop_first=True)

# Features & target
X = df.drop("Electric Range", axis=1)
y = df["Electric Range"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (better than linear regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)

#%%
import sys
print(sys.executable)
# %%
