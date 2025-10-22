import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
def load_wine_dataset():
    wine = fetch_ucirepo(id = 109)
    X = pd.DataFrame(wine.data.features)
    y = pd.DataFrame(wine.data.targets)

    X = X.to_numpy()
    y = y.to_numpy().ravel() 
    return X, y




