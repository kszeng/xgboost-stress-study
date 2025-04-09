import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split 

# exploratory trial run (https://www.datacamp.com/tutorial/xgboost-in-python)

warnings.filterwarnings("ignore")
diamonds = sns.load_dataset("diamonds")
diamonds.head()

# show shape of the practice dataset
print(diamonds.shape)

# for real-world data: spend much more time ecploring and visualizing dataset
# this is a 5-number summary that is built into sns.
print(diamonds.describe())

# extract the feature and target arrays
X, y = diamonds.drop('price', axis=1), diamonds[['price']]