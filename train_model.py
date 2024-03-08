import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from model import Condiitonal_gmms

data_path = r'data\all_user_data_dummy.csv'
df = pd.read_csv(data_path) # 12 condition columns
df = df.iloc[:,1:] 

# drop nan
df = df.dropna()

X = df.iloc[:, -12:]
Y = df.iloc[:, :-12]

# Fit the model
c_gmm = Condiitonal_gmms(n_components=5)
c_gmm.fit(X.values, Y.values)

# save the model
with open('c_gmm.pkl', 'wb') as f:
    pickle.dump(c_gmm, f)