
import numpy as np, pandas as pd
def read_mind_csv(path: str):
    df = pd.read_csv(path, index_col=0)
    labels = list(df.columns)
    mat = df.values.astype(np.float32)
    mat = (mat + mat.T)/2.0
    np.fill_diagonal(mat, 1.0)
    return mat, labels
