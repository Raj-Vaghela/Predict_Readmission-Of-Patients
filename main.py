import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('diabetic_data.csv', na_values='?' , low_memory=False)

print(df.describe())
print('\nshape of original data:',df.shape)
print('-'*163)

df.drop(columns=['encounter_id'], inplace=True)
missingValues = df.isna().sum()
missingValues = missingValues[missingValues>0]
missingPercentage = (missingValues/len(df))*100

missingInfo = pd.DataFrame({'Missing Values': missingValues, 'Missing Percentage': missingPercentage})
print(missingInfo)
print('\nShape after dropping encounter_id:',df.shape)
print('-'*163)

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

print('-'*163)
value_counts = df['readmitted'].value_counts()

# Print the occurrence of each unique value
print("Occurrence of each unique value in the column:")
for value, count in value_counts.items():
    print(f"{value}: {count}")
print('ratio :',90409/11357)
print('-'*163)
print('values in admission_type_id :')
print(df['admission_type_id'].unique())
print('-'*163)
print('-'*163)
print('value in discharge_disposition_id :')
print(df['discharge_disposition_id'].unique())
print('-'*163)
print('-'*163)
print('value in admission_source_id :')
print(df['admission_source_id'].unique())
print('-'*163)
# Print the occurrence of each unique value
print("Occurrence of each unique value in the column:")
for value, count in value_counts.items():
    print(f"{value}: {count}")
columns_to_drop_from_ProblemStatement = ['repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

col_with_over_90perc_MisVal = missingPercentage[missingPercentage>90].index.tolist()

columns_to_drop = columns_to_drop_from_ProblemStatement + col_with_over_90perc_MisVal

df.drop(columns=columns_to_drop, inplace=True)
df = df.dropna()

print('\nShape of data after dropping cols mentioned in Problem Statement and cols with more than 90% missing values and empty rows')
print(df.shape)
print('-'*163)
def identify_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

df_no_outliers = df.copy()

for column in df.select_dtypes(include=['int64', 'float64']).columns:
    if column != 'readmitted':
        outliers = identify_outliers_iqr(df[column])
        num_outliers = outliers.sum()
        df_no_outliers = df_no_outliers.loc[~outliers]
        print(f"Number of outliers by IQR in '{column}': {num_outliers}")
    print(df_no_outliers.shape)

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

print('Shape of Data after removing outliers: ',df_no_outliers.shape)
print('-'*163)

# Perform one-hot encoding for the admission_type_id , admission_source_id and  discharge_disposition_id column

# df_no_outliers = pd.get_dummies(df_no_outliers, columns=['admission_source_id'], prefix='admission_source_')
# # print(df_no_outliers.columns)
# print(df_no_outliers.shape)
#
# df_no_outliers = pd.get_dummies(df_no_outliers, columns=['discharge_disposition_id'], prefix='discharge_disposition_')
# # print(df_no_outliers.columns)
# print(df_no_outliers.shape)
#
# df_no_outliers = pd.get_dummies(df_no_outliers, columns=['admission_type_id'], prefix='admission_type_')
# # print(df_no_outliers.columns)
# print(df_no_outliers.shape)
#
# corr_matrix = df_no_outliers.corr()
#
# # Extract the correlation values with the 'readmitted' column
# admission_type_corr = corr_matrix['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('admission_type')]]
#
# # Print the correlation values
# print("Correlation of admission type columns with 'readmitted':")
# print(admission_type_corr)
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(admission_type_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
# plt.title("Correlation Heatmap of 'readmitted' with Admission Type")
# plt.xlabel("Admission Type")
# plt.ylabel("Correlation with 'readmitted'")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
#
# corr_matrix_1 = df_no_outliers.corr()
#
# # Extract the correlation values with the 'readmitted' column
# discharge_disposition_corr = corr_matrix_1['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('discharge_disposition')]]
# print("Correlation of discharge disposition columns with 'readmitted':")
# print(discharge_disposition_corr)
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(discharge_disposition_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
# plt.title("Correlation Heatmap of 'readmitted' with Discharge Disposition")
# plt.xlabel("Discharge Disposition")
# plt.ylabel("Correlation with 'readmitted'")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
# corr_matrix_2 = df_no_outliers.corr()
#
# # Extract the correlation values with the 'readmitted' column
# admission_source_corr = corr_matrix_2['readmitted'][df_no_outliers.columns[df_no_outliers.columns.str.startswith('admission_source')]]
# print("Correlation of admission source columns with 'readmitted':")
# print(admission_source_corr)
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(admission_source_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
# plt.title("Correlation Heatmap of 'readmitted' with Admission Source")
# plt.xlabel("Admission Source")
# plt.ylabel("Correlation with 'readmitted'")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

print('-'*163)
print(df_no_outliers.shape)

# admission_type_cols = [col for col in df_no_outliers.columns if col.startswith('admission_type')]
# df_no_outliers = df_no_outliers.drop(columns=admission_type_cols)
#
# # Drop columns starting with 'discharge_disposition'
# discharge_disposition_cols = [col for col in df_no_outliers.columns if col.startswith('discharge_disposition')]
# df_no_outliers = df_no_outliers.drop(columns=discharge_disposition_cols)
#
# # Drop columns starting with 'admission_source'
# admission_source_cols = [col for col in df_no_outliers.columns if col.startswith('admission_source')]
# df_no_outliers = df_no_outliers.drop(columns=admission_source_cols)
# 'number_emergency','number_outpatient'
col_to_normalize = ['number_inpatient','num_medications','num_procedures','num_lab_procedures','time_in_hospital','number_diagnoses']

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

df_no_outliers[col_to_normalize] = df_no_outliers[col_to_normalize].apply(min_max_scaling)

print(df_no_outliers.T.head(20))

# Assuming df is your DataFrame
# cols_to_drop = ['diag_1', 'diag_2', 'diag_3']  # Replace 'col1', 'col2', 'col3' with the column names you want to drop
# df_no_outliers = df_no_outliers.drop(columns=cols_to_drop)

print('Before Encoding shape:',df_no_outliers.shape)

categorical_cols_forLabelEncoding = ['diabetesMed','change','insulin','rosiglitazone','pioglitazone','glyburide','glipizide','metformin','A1Cresult','max_glu_serum','age','gender']
categorical_cols_forOneHotEncoding = ['diag_1','diag_2','diag_3','medical_specialty','payer_code','admission_type_id','discharge_disposition_id','admission_source_id','race']
# Perform one-hot encoding for each categorical column


# Apply label encoding to each set of columns
# for feature in categorical_cols_forLabelEncoding:
#     le = LabelEncoder()
#     df_no_outliers[feature] = le.fit_transform(df_no_outliers[feature])
#
#
# cat_cols = df_no_outliers[categorical_cols_forOneHotEncoding]
# encoder = OneHotEncoder()
# encoded_features = encoder.fit_transform(cat_cols)
# encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names(categorical_cols_forOneHotEncoding))
# df_no_outliers = pd.concat([df_no_outliers.drop(columns=categorical_cols_forOneHotEncoding), encoded_df], axis=1)

for feature in categorical_cols_forLabelEncoding:
    df_no_outliers[feature] = pd.factorize(df_no_outliers[feature])[0]

# Apply one-hot encoding to each set of columns separately
df_no_outliers = pd.get_dummies(df_no_outliers, columns=categorical_cols_forOneHotEncoding, drop_first=True)


print('After Encoding shape :',df_no_outliers.shape)
print('-'*163)



class_counts = df_no_outliers['readmitted'].value_counts()

# Plot the distribution of unique classes
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Unique Classes (Target Variable)')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.show()



col_to_move = df_no_outliers.pop('readmitted')

df_no_outliers['readmitted'] = col_to_move

# correlation_matrix = df_no_outliers.corr()
#
# plt.figure(figsize=(16, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix Heatmap')
# plt.show()




print('-'*163)



print('-'*163)

dup = df_no_outliers.duplicated(subset=['patient_nbr'], keep=False)
print(dup.sum())
print(df_no_outliers['patient_nbr'].nunique())

value_counts = df_no_outliers['patient_nbr'].value_counts()

# Filter values that occur only once
unique_values = value_counts[value_counts == 1]

# Count the number of unique values
num_unique_values = len(unique_values)

# Print the number of values occurring only once
print("Number of values occurring only once in the column:", num_unique_values)
print(dup.sum() + num_unique_values)

print('-'*163)

print(df_no_outliers['readmitted'].unique())

value_counts = df_no_outliers['readmitted'].value_counts()

# Print the occurrence of each unique value
print("Occurrence of each unique value in the column:")
for value, count in value_counts.items():
    print(f"{value}: {count}")
print('ratio : ', 23898/2857)
print('-'*163)
num_object_columns = len(df_no_outliers.select_dtypes(include=['object']).columns)

print("Number of columns with object datatype:", num_object_columns)
print('-'*163)
print('MODEL BUILDING')
print('-'*163)

print(df_no_outliers.shape)
#
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
#
# # Step 1: Split the data into features (X) and target variable (y)
# X = df_no_outliers.drop(columns=['readmitted'])
# y = df_no_outliers['readmitted']
#
# # Step 2: Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 3: Build the linear model and evaluate it using cross-validation
# model = LogisticRegression()
# cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
#
# print("Cross-validation F1 scores:", cv_scores)
# print("Mean F1 score:", cv_scores.mean())
#
# # Step 4: Train the model on the entire training set
# model.fit(X_train, y_train)
#
# # Step 5: Evaluate the model on the test set
# y_pred = model.predict(X_test)
# print("Classification Report on Test Set:")
# print(classification_report(y_test, y_pred))




from sklearn.model_selection import train_test_split, cross_val_score

# Split the data into features (X) and target variable (y)
X = df_no_outliers.drop(columns=['readmitted'])
y = df_no_outliers['readmitted']

# Split the data into training and test sets, stratifying by the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set - Class distribution:")
print("Zeros:", (y_train == 0).sum(), "Ones:", (y_train == 1).sum())

print("Test set - Class distribution:")
print("Zeros:", (y_test == 0).sum(), "Ones:", (y_test == 1).sum())

linear_model = LinearRegression()

# Train the model on the training set
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Convert predicted values to binary classifications (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)

print('-'*163)

# Compute accuracy
accuracy = (y_pred_binary == y_test).mean()
print("Accuracy:", accuracy)

# Compute precision
true_positives = ((y_pred_binary == 1) & (y_test == 1)).sum()
false_positives = ((y_pred_binary == 1) & (y_test == 0)).sum()
precision = true_positives / (true_positives + false_positives)
print("Precision:", precision)



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
print('-'*163)
# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print('-'*163)
# Compute Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)
print('-'*163)
# Compute R-squared (R2)
r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)
print('-'*163)

# Compute Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)
print('-'*163)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Convert predicted probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred_binary)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred_binary)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred_binary)
print("F1-score:", f1)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC score:", roc_auc)

#---------------------------------------------------------------------------------------------------------------------
# model = LogisticRegression()
# cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
#
# print("\n\nCross-validation F1 scores:", cv_scores)
# print("Mean F1 score:", cv_scores.mean())
#
# # Step 4: Train the model on the entire training set
# model.fit(X_train, y_train)
#
# # Step 5: Evaluate the model on the test set
# y_pred = model.predict(X_test)
# print("Classification Report on Test Set:")
# print(classification_report(y_test, y_pred))

#---------------------------------------------------------------------------------------------------------------------
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#
# # Build a logistic regression model
# model = LogisticRegression()
#
# # Train the model on the resampled data
# model.fit(X_train_resampled, y_train_resampled)
#
# # Evaluate the model on the test set
# y_pred = model.predict(X_test)
# print("Classification Report on Test Set:")
# print(classification_report(y_test, y_pred))