import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_features(df, input_features, x_feats, y_feats):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scalerx = StandardScaler()
    scalery = StandardScaler()

    encoded_categorical = encoder.fit_transform(df[input_features])
    scaled_x = scalerx.fit_transform(df[x_feats])
    scaled_y = scalery.fit_transform(df[y_feats])

    # combine features
    X = np.hstack((encoded_categorical, scaled_x))
    y = scaled_y

    return X, y, encoder, scalerx, scalery
