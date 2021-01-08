import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        df = df.where(df.notnull(), 'Nan')
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df


data = pd.read_csv("data/test.csv")

df = dummyEncode(data)

df = df/df.max().astype(np.float64)


#df.apply(lambda x: x/x.max(), axis=0)

pd.DataFrame.to_csv(df, "data/processed.csv")