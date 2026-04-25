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

#Clean Alternative Fuel Vehicle (CAFV) Eligibility
print('Clean Alternative Fuel Vehicle :',df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].isnull().sum())
#%%

print(df['County'].value_counts().plot(kind='bar'))
#%%
 
df.isnull().sum()

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
df['Make'].value_counts().head(10).plot(kind='bar')
#%%
sns.histplot(df['Electric Range'], bins=30)
plt.title("Electric Range Distribution")
plt.show()
#%%
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.tight_layout()
plt.show()

