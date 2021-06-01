import numpy
import pandas as pd
from numpy import argmax

from Model import GlobalVariables
from Model.GlobalVariables import raw_data_file, output, inputfeatures, scaler


# Read the raw data into data frame df
# Transform features in df to X and output to y
from Model.DeeplearningModel import SaveDictToFile


def ReadData():
    df = pd.read_csv(raw_data_file)
    df.dropna(inplace=True)
    df = df[(df.MIXA_PASTEUR_STATE == 0) | (df.MIXA_PASTEUR_STATE == 1)]

    # Temprature 800 => 80.0
    df.iloc[:, 3] /= 10
    df.iloc[:, 4] /= 10

    X = df[inputfeatures]
    y = df[output]
    print(len(X))
    return df, X, y


# Scale data
def ScaleData(X, y):
    X_transformed = scaler.fit_transform(X)

    # Create dictionaries for labels in y
    output_set = y.INSP.unique()
    int_to_name = {k: w for k, w in enumerate(output_set)}
    name_to_int = {w: k for k, w in int_to_name.items()}

    # encode values of y
    y_encoded = []
    for i in range(0, len(y)):
        y_encoded.append(one_hot_encode(name_to_int, y.iloc[i, 0]))

    y_encoded = numpy.asarray(y_encoded)
    return X_transformed, y_encoded, name_to_int, int_to_name


# one hot encode sequence
def one_hot_encode(name_to_int_dict, value):
    vector = [0 for _ in range(len(name_to_int_dict))]
    key = name_to_int_dict[value]
    vector[key] = 1
    return vector


def one_hot_decode(encoded_seq):
    result = [argmax(vector) for vector in encoded_seq]
    return result
