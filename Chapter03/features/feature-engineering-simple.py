from sklearn import preprocessing

data = [['Bleach'], ['Cereal'], ['Toilet Roll']]

if __name__=="__main__":
    ordinal_enc = preprocessing.OrdinalEncoder()
    ordinal_enc.fit(data)
    print(ordinal_enc.transform(data))

    onehot_enc = preprocessing.OneHotEncoder()
    onehot_enc.fit(data)
    print(onehot_enc.transform(data).toarray())


