import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.info())
print(df.describe())
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)

