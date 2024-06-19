import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('house_pricing/data/housing.csv')
df.dropna(inplace=True)
train, test= train_test_split(df, test_size=0.2)
X_train = train.drop(columns='median_house_value')
y_train = train['median_house_value']

train['total_rooms'] = np.log(train['total_rooms'] + 1)

