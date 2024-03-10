import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetic_data.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.isnull()
      .value_counts())
