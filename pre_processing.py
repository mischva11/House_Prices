import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
        le = LabelEncoder()
        df = df.where(df.notnull(), 'Nan')
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df


data = pd.read_csv("data/train.csv")

df = dummyEncode(data)

for column in df:
    data[column] = df[column]

data = data/data.max().astype(np.float64)

#pd.DataFrame.to_csv(data, "data/processed.csv")

test = data.columns[data.isnull().any()]

print(test)