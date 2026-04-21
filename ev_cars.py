#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\iec\portfolio2\66634-ev-data\Electric_Vehicle_Population_Data.csv')
#%%
print(df.head(50))
#%%
df.head()
df.shape
df.columns
df.info()
df.describe()
#%%
# checking for null
#VIN (1-10)
print('VIN (1-10):',df['VIN (1-10)'].isnull().sum)
#%% 
#County
print('County:',df['County'].isnull().sum())
#%%
#City
print('City:',df['City'].isnull().sum())
#%%
#State
print('State:',df['State'].isnull().sum())
#%%
#Postal Code
print('Postal Code:',df['Postal Code'].isnull().sum())
#%%
#Model Year
print('Model Year:',df['Model Year'].isnull().sum())
#%%
#Make
print('Make:',df['Make'].isnull().sum())
#%%
#Model
print('Model:',df['Model'].isnull().sum())
#%%
#Electric Vehicle Type
print('Electric Vehicle Type:',df['Electric Vehicle Type'].isnull().sum())
#%%
#Clean Alternative Fuel Vehicle (CAFV) Eligibility
print('Clean Alternative Fuel Vehicle :',df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].isnull().sum())
#%%
#Electric Range
print('Electric Range:',df['Electric Range'].isnull().sum())
#%%
#Base MSRP
print('Base MSRP:',df['Base MSRP'].isnull().sum())
#%%
#Legislative District
print('Legislative District:',df['Legislative District'].isnull().sum())
#%%
#DOL Vehicle ID
print('DOL Vehicle ID:',df['DOL Vehicle ID'].isnull().sum())
#%%
#Vehicle Location
print('Vehicle Location:',df['Vehicle Location'].isnull().sum())
#%%
#Electric Utility
print('Electric Utility:',df['Electric Utility'].isnull().sum())
#%%
#2020 Census Tract
print('2020 Census Tract:',df['2020 Census Tract'].isnull().sum())
#%%
print(df['County'].value_counts().plot(kind='bar'))
#%%
df['Make'].value_counts().plot(kind='pie')
print(df['Model Year'].value_counts())
plt.show()
#%%
df['County'].value_counts().head(10).plot(kind='bar')
plt.title("Top Counties by EV Count")
plt.xlabel("County")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=45)
plt.tight_layout()
#%%
df['Model Year'].value_counts().head(10).plot(kind='bar')
plt.title("Top model of EV cars ")
plt.xlabel("Models")
plt.ylabel("Numbers of Cars")
plt.tight_layout()
plt.show()
#%%
df['Model Year'].value_counts().sort_index().plot()
plt.title("EV Registrations by Model Year")
plt.show()
#%%
sns.histplot(df['Electric Range'], bins=30)
plt.title("Electric Range Distribution")
plt.show()
#%%
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.tight_layout()
plt.show()