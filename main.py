import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
missingValues = df.isna().sum()
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
df = df.dropna()

print('\nShape of data after dropping cols mentioned in Problem Statement and cols with more than 90% missing values and empty rows')
print(df.shape)
print('-'*163)
# Define a function to identify outliers using Z-score method
def identify_outliers_zscore(data, threshold=4):
    z_scores = zscore(data)
    outliers = (abs(z_scores) > threshold)
    return outliers
def identify_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# df = df.copy()
# Loop through each column in the DataFrame
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        outliers = identify_outliers_zscore(df[column])
        num_outliers = outliers.sum()
        print(f"Number of outliers by Z-score in '{column}': {num_outliers}")

print('-'*163)
# for column in df.columns:
#     if df[column].dtype in ['int64', 'float64']:
#         outliers = identify_outliers_iqr(df[column])
#         outliers = outliers.sum()
#         df = df[~outliers]
#         print(f"Number of outliers by IQR in '{column}': {outliers}")
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    if column !='readmitted':
        outliers = identify_outliers_iqr(df[column])
        num_outliers = outliers.sum()
        # x = outliers.any(axis=0)
        df_no_outliers = df[~outliers]
        print(f"Number of outliers by IQR in '{column}': {num_outliers}")
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

print(df_no_outliers.shape)
print('-'*163)

# unique_values = df['payer_code'].unique()
# unique_values_count = df['payer_code'].nunique()
# print(f"Number of unique values: {unique_values_count}")
# print("Different unique values occurring in the 'payer_code' column:")
# for value in unique_values:
#     print(value)

admission_type_mapping ={
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'NULL',
    7: 'Trauma Center',
    8: 'Not Mapped'
}


print(df_no_outliers['admission_type_id'].unique())
print(df_no_outliers['admission_type_id'].value_counts())
print(df_no_outliers['admission_type_id'].isnull().sum())
# df['admission_type_id'] = df['admission_type_id'].astype(str)

df_no_outliers['admission_type_id'] = df_no_outliers['admission_type_id'].map(admission_type_mapping)

print(df_no_outliers['admission_type_id'].unique())
print(df_no_outliers['admission_type_id'].value_counts())
print(df_no_outliers['admission_type_id'].isnull().sum())

discharge_disposition_id_mapping = {
    1: 'Discharged to home',
    2: 'Discharged/transferred to another short term hospital',
    3: 'Discharged/transferred to SNF',
    4: 'Discharged/transferred to ICF',
    5: 'Discharged/transferred to another type of inpatient care institution',
    6: 'Discharged/transferred to home with home health service',
    7: 'Left AMA',
    8: 'Discharged/transferred to home under care of Home IV provider',
    9: 'Admitted as an inpatient to this hospital',
    10: 'Neonate discharged to another hospital for neonatal aftercare',
    11: 'Expired',
    12: 'Still patient or expected to return for outpatient services',
    13: 'Hospice / home',
    14: 'Hospice / medical facility',
    15: 'Discharged/transferred within this institution to Medicare approved swing bed',
    16: 'Discharged/transferred/referred another institution for outpatient services',
    17: 'Discharged/transferred/referred to this institution for outpatient services',
    18: 'NULL',
    19: 'Expired at home. Medicaid only, hospice.',
    20: 'Expired in a medical facility. Medicaid only, hospice.',
    21: 'Expired, place unknown. Medicaid only, hospice.',
    22: 'Discharged/transferred to another rehab fac including rehab units of a hospital .',
    23: 'Discharged/transferred to a long term care hospital.',
    24: 'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    25: 'Not Mapped',
    26: 'Unknown/Invalid',
    30: 'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    27: 'Discharged/transferred to a federal health care facility.',
    28: 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    29: 'Discharged/transferred to a Critical Access Hospital (CAH).',
}
# unique_values = df['admission_type_id'].unique()
# unique_values_count = df['admission_type_id'].nunique()
# print(f"Number of unique values: {unique_values_count}")
# print("Different unique values occurring in the 'admission_type_id' column:")
# for value in unique_values:
#     print(value)

df_no_outliers['discharge_disposition_id'] = df_no_outliers['discharge_disposition_id'].map(discharge_disposition_id_mapping)

admisssion_source_id_mapping = {
    1: 'Physician Referral',
    2: 'Clinic Referral',
    3: 'HMO Referral',
    4: 'Transfer from a hospital',
    5: 'Transfer from a Skilled Nursing Facility (SNF)',
    6: 'Transfer from another health care facility',
    7: 'Emergency Room',
    8: 'Court/Law Enforcement',
    9: 'Not Available',
    10: 'Transfer from critial access hospital',
    11: 'Normal Delivery',
    12: 'Premature Delivery',
    13: 'Sick Baby',
    14: 'Extramural Birth',
    15: 'Not Available',
    17: 'NULL',
    18: 'Transfer From Another Home Health Agency',
    19: 'Readmission to Same Home Health Agency',
    20: 'Not Mapped',
    21: 'Unknown/Invalid',
    22: 'Transfer from hospital inpt/same fac reslt in a sep claim',
    23: 'Born inside this hospital',
    24: 'Born outside this hospital',
    25: 'Transfer from Ambulatory Surgery Center',
    26: 'Transfer from Hospice',
}

df_no_outliers = pd.get_dummies(df_no_outliers, columns=['admission_source_id'], prefix='admission_source_')

print(df_no_outliers.columns)
print(df_no_outliers.shape)



df_no_outliers = pd.get_dummies(df_no_outliers, columns=['discharge_disposition_id'], prefix='discharge_disposition_')

print(df_no_outliers.columns)
print(df_no_outliers.shape)




# Perform one-hot encoding for the 'admission_type_id' column
df_no_outliers = pd.get_dummies(df_no_outliers, columns=['admission_type_id'], prefix='admission_type_')
print(df_no_outliers.columns)
print(df_no_outliers.shape)

corr_matrix = df_no_outliers.corr()

# Extract the correlation values with the 'readmitted' column
admission_type_corr = corr_matrix['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('admission_type')]]

# Print the correlation values
print("Correlation of admission type columns with 'readmitted':")
print(admission_type_corr)

plt.figure(figsize=(10, 6))
sns.heatmap(admission_type_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title("Correlation Heatmap of 'readmitted' with Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Correlation with 'readmitted'")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


corr_matrix_1 = df_no_outliers.corr()

# Extract the correlation values with the 'readmitted' column
discharge_disposition_corr = corr_matrix_1['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('discharge_disposition')]]
print("Correlation of discharge disposition columns with 'readmitted':")
print(discharge_disposition_corr)

plt.figure(figsize=(10, 6))
sns.heatmap(discharge_disposition_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title("Correlation Heatmap of 'readmitted' with Discharge Disposition")
plt.xlabel("Discharge Disposition")
plt.ylabel("Correlation with 'readmitted'")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

corr_matrix_2 = df_no_outliers.corr()

# Extract the correlation values with the 'readmitted' column
admission_source_corr = corr_matrix_2['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('admission_source')]]
print("Correlation of admission source columns with 'readmitted':")
print(admission_source_corr)

plt.figure(figsize=(10, 6))
sns.heatmap(admission_source_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title("Correlation Heatmap of 'readmitted' with Admission Source")
plt.xlabel("Admission Source")
plt.ylabel("Correlation with 'readmitted'")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

admission_type_cols = [col for col in df_no_outliers.columns if col.startswith('admission_type')]
df_no_outliers = df_no_outliers.drop(columns=admission_type_cols)

# Drop columns starting with 'discharge_disposition'
discharge_disposition_cols = [col for col in df_no_outliers.columns if col.startswith('discharge_disposition')]
df_no_outliers = df_no_outliers.drop(columns=discharge_disposition_cols)

# Drop columns starting with 'admission_source'
admission_source_cols = [col for col in df_no_outliers.columns if col.startswith('admission_source')]
df_no_outliers = df_no_outliers.drop(columns=admission_source_cols)

col_to_normalize = ['number_inpatient','number_emergency','number_outpatient','num_medications','num_procedures','num_lab_procedures','time_in_hospital','number_diagnoses']

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

df_no_outliers[col_to_normalize] = df_no_outliers[col_to_normalize].apply(min_max_scaling)
print(df_no_outliers)


class_counts = df_no_outliers['readmitted'].value_counts()

# Plot the distribution of unique classes
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Unique Classes (Target Variable)')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.show()


age_intervals = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

# Group data by age intervals and count readmitted cases
readmitted_counts = df_no_outliers.groupby('age')['readmitted'].sum()

# Convert age intervals to corresponding labels
readmitted_counts.index = age_labels

# Plot the count of readmitted cases against age intervals
plt.figure(figsize=(10, 6))
readmitted_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Age Intervals')
plt.ylabel('Count of Readmitted Cases')
plt.title('Count of Readmitted Cases by Age Intervals')
plt.xticks(rotation=45, ha='right')
plt.show()


target_counts = df_no_outliers.groupby('num_medications')['readmitted'].value_counts().unstack(fill_value=0)

plt.figure(figsize=(14,4))
bars = target_counts.plot(kind='bar', stacked=True, width=0.8, align='center')  # Set align='center' to align bars to tick labels

# Get the current positions of the bars
positions = np.arange(len(target_counts))

# Manually adjust the positions of the bars
plt.xticks(positions + 0.2 * (len(target_counts.columns) - 1) / 2, target_counts.index)
plt.xlabel('Number of Medications')
plt.ylabel('Count')
plt.title('Count of Target Variable against Number of Medications (Normalized)')
plt.legend(title='Readmitted', loc='upper left', labels=['Not Readmitted', 'Readmitted'])
plt.xticks(rotation=90)
plt.show()



target_counts = df_no_outliers.groupby('num_medications')['readmitted'].value_counts().unstack(fill_value=0)

plt.figure(figsize=(14, 6))
target_counts.plot(kind='line', marker='o', markersize=8)
plt.xlabel('Number of Medications')
plt.ylabel('Count')
plt.title('Count of Target Variable against Number of Medications (Normalized)')
plt.legend(title='Readmitted', loc='upper right', labels=['Not Readmitted', 'Readmitted'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# sns.pairplot(df_no_outliers)
# plt.title('Scatter Matrix Plot (Pairplot)')
# plt.show()

col_to_move = df_no_outliers.pop('readmitted')

df_no_outliers['readmitted'] = col_to_move

correlation_matrix = df_no_outliers.corr()

plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# Drop columns starting with 'admission_type'


# Print dtype of each column
for column in df_no_outliers.columns:
    print(f"{column}: {df_no_outliers[column].dtype}")

