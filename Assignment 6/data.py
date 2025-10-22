import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_preprocess_data():
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features)
    y= pd.DataFrame(iris.data.targets)

    y = y.applymap(lambda val :val.replace("Iris-", ""))

    X=X.to_numpy()
    y=y.to_numpy().ravel()

    return X, y

X, y = load_preprocess_data()