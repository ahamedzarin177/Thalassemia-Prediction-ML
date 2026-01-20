def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    X = df.drop('Result', axis=1)
    y = df['Result']
    return X, y

