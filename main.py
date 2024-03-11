import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetic_data.csv', na_values='?' , low_memory=False)

# withQM = pd.read_csv('diabetic_data.csv')
# print(withQM.describe())
# print(withQM.shape)
# print(len(withQM))
print(df.describe())
print('\nshape of original data:',df.shape)
print('-'*163)
# print(df.isnull().sum())

# print(f"The count of '?' :") #Checks for ?s in the dataset's columns
# for col in withQM.columns:
#     value = withQM[col].value_counts()
#     count = value.get('?',0)
#     percentage = count / len(withQM) * 100
#
#
#     if count > 0:
#         print(f"\t column '{col}' is: {count} (that is {percentage:.2f} )")
#         match_found = True
# if not match_found:
#     print(f"\t 'Never occured'")
df.drop(columns=['encounter_id'], inplace=True)
missingValues = df.isnull().sum()
missingValues = missingValues[missingValues>0]
missingPercentage = (missingValues/len(df))*100


missingInfo = pd.DataFrame({'Missing Values': missingValues, 'Missing Percentage': missingPercentage})
print(missingInfo)
print('\nShape after dropping encounter_id:',df.shape)
print('-'*163)

# code to display all unique values in a column
# unique_values = df['readmitted'].unique()
#
# print("Different values occurring in the column:")
# for value in unique_values:
#     print(value)

# for column_name in df.columns:
#     unique_values = df[column_name].unique()
#     print(f"Different values occurring in column '{column_name}':")
#     for value in unique_values:
#         print(value)
#     print()

unique_values = df['readmitted'].unique()
print("\nDifferent values occurring in the 'readmitted' column:")
for value in unique_values:
    print(value)

df['readmitted'] = df['readmitted'].replace({'<30':1, '>30':0, 'NO':0}, )

unique_values = df['readmitted'].unique()
print("\nDifferent New values occurring in the 'readmitted' column:")
for value in unique_values:
    print(value)

columns_to_drop_from_ProblemStatement = ['repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

col_with_over_90perc_MisVal = missingPercentage[missingPercentage>90].index.tolist()
# print(col_with_over_90perc_MisVal)

columns_to_drop = columns_to_drop_from_ProblemStatement + col_with_over_90perc_MisVal
# print(columns_to_drop)

df.drop(columns=columns_to_drop, inplace=True)
print('\nShape of data after dropping cols mentioned in Problem Statement and cols with more than 90% missing values')
print(df.shape)
print('-'*163)

# unique_values = df['max_glu_serum'].unique()
# print("\nDifferent New values occurring in the 'max_glu_serum' column:")
# for value in unique_values:
#     print(value)


