import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(dataset_path):
    frame = pd.read_csv(dataset_path)

    categorical_features = ["Geography", "Gender"]

    frame = pd.get_dummies(frame, columns=categorical_features)

    frame.drop('RowNumber', axis=1, inplace=True)
    frame.drop('CustomerId', axis=1, inplace=True)
    frame.drop('Surname', axis=1, inplace=True)

    x = frame.iloc[:, np.r_[0:8, 10:13]].values  # using 10:13 instead of 9:14 to avoid dummy variable trap
    y = frame.iloc[:, 8].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # normalization
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    return x_train, x_test, y_train, y_test







