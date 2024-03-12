import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, skew, normaltest

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
unique_values_count = df['readmitted'].nunique()
print(f"Number of unique values in the 'readmitted' column: {unique_values_count}")
print("Different values occurring in the 'readmitted' column:")
for value in unique_values:
    print(value)

df['readmitted'] = df['readmitted'].replace({'<30':1, '>30':0, 'NO':0}, )

unique_values = df['readmitted'].unique()
unique_values_count = df['readmitted'].nunique()
print(f"\nNumber of unique values in the 'readmitted' column: {unique_values_count}")
print("Different New values occurring in the 'readmitted' column:")
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
# Define a function to identify outliers using Z-score method
def identify_outliers_zscore(data, threshold=4):
    z_scores = zscore(data)
    outliers = (abs(z_scores) > threshold)
    return outliers
def count_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers.sum()


# Loop through each column in the DataFrame
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        outliers = identify_outliers_zscore(df[column])
        num_outliers = outliers.sum()
        print(f"Number of outliers by Z-score in '{column}': {num_outliers}")

print('-'*163)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        outliers = count_outliers_iqr(df[column])
        num_outliers = outliers.sum()
        print(f"Number of outliers by IQR in '{column}': {outliers}")
print('-'*163)

# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#
# fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4 * len(numeric_cols))) # Create subplots with 2 columns
#
# for i, col in enumerate(numeric_cols): # Plot histograms and density plots for each numeric column and perform normality test
#
#     axes[i, 0].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#     axes[i, 0].set_title('Histogram of ' + col)
#     axes[i, 0].set_xlabel('Values')
#     axes[i, 0].set_ylabel('Frequency')
#     axes[i, 0].grid(True)
#
#     df[col].plot(kind='density', ax=axes[i, 1], color='skyblue')
#     axes[i, 1].set_title('Density Plot of ' + col)
#     axes[i, 1].set_xlabel('Values')
#     axes[i, 1].set_ylabel('Density')
#     axes[i, 1].grid(True)
#
#     skewness = skew(df[col])     # Calculate skewness
#     normal_test_result = normaltest(df[col]) # Perform normality test
#
#     print(f"Column: {col}")
#     print(f"Skewness: {skewness}")
#     print(f"Normality test p-value: {normal_test_result.pvalue}")
#
#     # Determine skewness interpretation
#     if abs(skewness) < 0.5:
#         print("The data is approximately symmetric (close to normally distributed).")
#     elif skewness < -0.5:
#         print("The data is left-skewed.")
#     else:
#         print("The data is right-skewed.")
#
#     # Determine normality test interpretation
#     if normal_test_result.pvalue < 0.05:
#         print("The data is not normally distributed (NOTE: use IQR METHOD to determine outliers) (reject the null hypothesis).")
#     else:
#         print("The data is normally distributed (NOTE: use Z-score METHOD to determine outliers) (fail to reject the null hypothesis).")
#
#     print('-'*163)
#     plt.subplots_adjust(hspace=0.5)     # Add some space between subplots
# plt.show()

print(df.shape)
print('-'*163)

# unique_values = df['payer_code'].unique()
# unique_values_count = df['payer_code'].nunique()
# print(f"Number of unique values: {unique_values_count}")
# print("Different unique values occurring in the 'payer_code' column:")
# for value in unique_values:
#     print(value)



