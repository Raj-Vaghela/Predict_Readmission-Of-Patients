import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv('diabetic_data.csv', na_values=['?'] , low_memory=False)

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
print('ratio  :',value_counts[0]/(value_counts[1]))
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

for column in df_no_outliers.select_dtypes(include=['int64', 'float64']).columns:
    if column != 'readmitted':
        outliers = identify_outliers_iqr(df[column])
        num_outliers = outliers.sum()
        df_no_outliers = df_no_outliers.loc[~outliers]
        print(f"Number of outliers by IQR in '{column}': {num_outliers}")
    print('removed outliers datasize):',df_no_outliers.shape)

print('-'*163)
#
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
for column in df_no_outliers.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Summary statistics for column '{column}':")
    print(df_no_outliers[column].describe())
    print('Variance : ', df_no_outliers[column].var())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Histogram using Seaborn
    sns.histplot(data=df_no_outliers, x=column, ax=axes[0])
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f"Histogram of column '{column}'")

    # Bar plot using Seaborn
    value_counts = df_no_outliers[column].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[1])
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f"Bar Plot of column '{column}'")

    # Box plot using Seaborn
    sns.boxplot(data=df_no_outliers, x=column, ax=axes[2])
    axes[2].set_xlabel(column)
    axes[2].set_ylabel('Values')
    axes[2].set_title(f"Box Plot of column '{column}'")

    plt.tight_layout()
    plt.show()

print('Shape of Data after removing all outliers: ',df_no_outliers.shape)
print('-'*163)

col_to_normalize = ['number_inpatient','num_medications','num_procedures','num_lab_procedures','time_in_hospital','number_diagnoses']

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

df_no_outliers[col_to_normalize] = df_no_outliers[col_to_normalize].apply(min_max_scaling)

print(df_no_outliers.T.head(20))


print('Before Encoding shape:',df_no_outliers.shape)

categorical_cols_forLabelEncoding = ['diabetesMed','change','insulin','rosiglitazone','pioglitazone','glyburide','glipizide','metformin',
                                     'A1Cresult','max_glu_serum',
                                     'age','gender']
categorical_cols_forOneHotEncoding = ['diag_1','diag_2','diag_3','medical_specialty','payer_code','admission_type_id','discharge_disposition_id','admission_source_id','race']

# Label encoding
for feature in categorical_cols_forLabelEncoding:
    df_no_outliers[feature] = pd.factorize(df_no_outliers[feature])[0]

# One-hot encoding
df_no_outliers = pd.get_dummies(df_no_outliers, columns=categorical_cols_forOneHotEncoding, drop_first=True)


print('After Encoding shape :',df_no_outliers.shape)
print('-'*163)




col_to_move = df_no_outliers.pop('readmitted')
df_no_outliers['readmitted'] = col_to_move

print('-'*163)



value_counts = df_no_outliers['readmitted'].value_counts()

# Print the occurrence of each unique value
print("Occurrence of each unique value in the column:")
for value, count in value_counts.items():
    print(f"{value}: {count}")
print('ratio : ', value_counts[0]/(value_counts[1]))


print('-'*163)
print('MODEL BUILDING')
print('-'*163)
#
# # print(df_no_outliers.shape)
# #
# #
# #
# # # Split the data into features (X) and target variable (y)
# # X = df_no_outliers.drop(columns=['readmitted'])
# # y = df_no_outliers['readmitted']
# #
# # # Split the data into training and test sets, stratifying by the target variable
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# #
# # print("Training set - Class distribution:")
# # print("Zeros:", (y_train == 0).sum(), "Ones:", (y_train == 1).sum())
# #
# # print("Test set - Class distribution:")
# # print("Zeros:", (y_test == 0).sum(), "Ones:", (y_test == 1).sum())
# #
# # linear_model = LinearRegression()
# #
# # # Train the model on the training set
# # linear_model.fit(X_train, y_train)
# #
# # # Make predictions on the test set
# # y_pred = linear_model.predict(X_test)
# #
# # # Evaluate the model using Mean Squared Error
# # mse = sk.metrics.mean_squared_error(y_test, y_pred)
# # print("Mean Squared Error:", mse)
# #
# #
# # # Convert predicted values to binary classifications (0 or 1)
# # y_pred_binary = (y_pred >= 0.5).astype(int)
# #
# # print('-'*163)
# #
# # # Compute accuracy
# # accuracy = (y_pred_binary == y_test).mean()
# # print("Accuracy:", accuracy)
# #
# # # Compute precision
# # true_positives = ((y_pred_binary == 1) & (y_test == 1)).sum()
# # false_positives = ((y_pred_binary == 1) & (y_test == 0)).sum()
# # precision = true_positives / (true_positives + false_positives)
# # print("Precision:", precision)
# #
# #
# #
# #
# # print('-'*163)
# # mae = sk.metrics.mean_absolute_error(y_test, y_pred)
# # print("Mean Absolute Error (MAE):", mae)
# # rmse = sk.metrics.mean_squared_error(y_test, y_pred, squared=False)
# # print("Root Mean Squared Error (RMSE):", rmse)
# # r2 = sk.metrics.r2_score(y_test, y_pred)
# # print("R-squared (R2):", r2)
# # mape = sk.metrics.mean_absolute_percentage_error(y_test, y_pred)
# # print("Mean Absolute Percentage Error (MAPE):", mape)
# # print('-'*163)
# #
# # # Convert predicted probabilities to binary predictions (0 or 1)
# # y_pred_binary = (y_pred >= 0.5).astype(int)
# #
# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred_binary)
# # print("Accuracy:", accuracy)
# #
# # # Calculate precision
# # precision = precision_score(y_test, y_pred_binary)
# # print("Precision:", precision)
# #
# # # Calculate recall
# # recall = recall_score(y_test, y_pred_binary)
# # print("Recall:", recall)
# #
# # # Calculate F1-score
# # f1 = f1_score(y_test, y_pred_binary)
# # print("F1-score:", f1)
# #
# # # Calculate ROC AUC score
# # roc_auc = roc_auc_score(y_test, y_pred)
# # print("ROC AUC score:", roc_auc)
# # #---------------------------------------------------------------------------------------------------------------------
# # # oversampling
# # #---------------------------------------------------------------------------------------------------------------------
# #
# # # Apply oversampling to the training set
# # X_train_resampled, y_train_resampled = imb.over_sampling.RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
# #
# # # Train the model on the resampled training set
# # linear_model = LinearRegression()
# # linear_model.fit(X_train_resampled, y_train_resampled)
# #
# # # Make predictions on the test set
# # y_pred = linear_model.predict(X_test)
# #
# # print('-'*163)
# # print('AFTER OVERSAMPLING')
# # print('-'*163)
# #
# # # Evaluate the model using Mean Squared Error
# # mse = sk.metrics.mean_squared_error(y_test, y_pred)
# # print("Mean Squared Error:", mse)
# #
# # # Convert predicted values to binary classifications (0 or 1)
# # y_pred_binary = (y_pred >= 0.5).astype(int)
# #
# # # Compute accuracy
# # accuracy = accuracy_score(y_test, y_pred_binary)
# # print("Accuracy:", accuracy)
# #
# # # Compute precision
# # precision = precision_score(y_test, y_pred_binary)
# # print("Precision:", precision)
# #
# # # Compute recall
# # recall = recall_score(y_test, y_pred_binary)
# # print("Recall:", recall)
# #
# # # Compute F1-score
# # f1 = f1_score(y_test, y_pred_binary)
# # print("F1-score:", f1)
# #
# # # Compute ROC AUC score
# # roc_auc = roc_auc_score(y_test, y_pred)
# # print("ROC AUC score:", roc_auc)
# #
# # #---------------------------------------------------------------------------------------------------------------------
# # #AFTER UNDERSAMPLING
# # #---------------------------------------------------------------------------------------------------------------------
# #
# # # Apply undersampling to the training set
# # X_train_resampled, y_train_resampled = imb.under_sampling.RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
# #
# # # Train the model on the resampled training set
# # linear_model = LinearRegression()
# # linear_model.fit(X_train_resampled, y_train_resampled)
# #
# # # Make predictions on the test set
# # y_pred = linear_model.predict(X_test)
# #
# #
# # print('-'*163)
# # print('AFTER UNDERSAMPLING')
# # print('-'*163)
# #
# # # Evaluate the model using Mean Squared Error
# # mse = mean_squared_error(y_test, y_pred)
# # print("Mean Squared Error:", mse)
# #
# # # Convert predicted values to binary classifications (0 or 1)
# # y_pred_binary = (y_pred >= 0.5).astype(int)
# #
# # # Compute accuracy
# # accuracy = accuracy_score(y_test, y_pred_binary)
# # print("Accuracy:", accuracy)
# #
# # # Compute precision
# # precision = precision_score(y_test, y_pred_binary)
# # print("Precision:", precision)
# #
# # # Compute recall
# # recall = recall_score(y_test, y_pred_binary)
# # print("Recall:", recall)
# #
# # # Compute F1-score
# # f1 = f1_score(y_test, y_pred_binary)
# # print("F1-score:", f1)
# #
# # # Compute ROC AUC score
# # roc_auc = roc_auc_score(y_test, y_pred)
# # print("ROC AUC score:", roc_auc)
#
# # #---------------------------------------------------------------------------------------------------------------------
# # #RANDOM FOREST
# # #---------------------------------------------------------------------------------------------------------------------
# #
# #
# # print('-'*163)
# # print('Random Forest')
# # print('-'*163)
# #
# #
# # data = df_no_outliers.copy()
# #
# # X = data.drop('readmitted', axis=1)
# # y = data['readmitted']
# #
# # sc_X = sk.preprocessing.StandardScaler()
# # X_s = sc_X.fit_transform(X)
# #
# # X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
# #
# # model = sk.ensemble.RandomForestClassifier()
# # model.fit(X_train, y_train)
# #
# # y_pred = model.predict(X_test)
# # print(f'Accuracy: {accuracy_score(y_test, y_pred):%}', )
# # print(f'Precision: { precision_score(y_test, y_pred, zero_division=1):%}',)
# # print(f'Recall: { recall_score(y_test, y_pred):%}',)
# # print(f'F1 Score: { f1_score(y_test, y_pred):%}',)
# #
# #
# # X_resampled, y_resampled = imb.over_sampling.SMOTE(random_state=42).fit_resample(X_s, y)
# #
# # X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# #
# # model_balance  = sk.ensemble.RandomForestClassifier()
# # model_balance.fit(X_train, y_train)
# #
# # y_pred_balanced = model_balance.predict(X_test)
# # print(f'Balanced Data - Accuracy: {accuracy_score(y_test, y_pred_balanced):%}', )
# # print(f'Balanced Data - Precision: {precision_score(y_test, y_pred_balanced,zero_division=1):%}', )
# # print(f'Balanced Data - Recall: {recall_score(y_test, y_pred_balanced):%}', )
# # print(f'Balanced Data - F1 Score: {f1_score(y_test, y_pred_balanced):%}', )
# #

#---------------------------------------------------------------------------------------------------------------------
#LOGISTIC REGRESSION
#---------------------------------------------------------------------------------------------------------------------
print('-'*163)
print('LOGISTIC REGRESSION')
print('-'*163)

