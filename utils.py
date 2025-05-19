import numpy as np

from sklearn.preprocessing import MinMaxScaler


def pad(RICR, pad_width):

    
    RICR = np.pad(RICR, ((0, 0), (0, pad_width), (0, 0)), constant_values=0)
    return RICR

def minmax(RICR):

    
    dim0, dim1, dim2 = RICR.shape
    scaler = MinMaxScaler()
    RICR_ = np.transpose(RICR, (1, 0, 2))
    RICR_ = RICR_.reshape(dim1, dim0*dim2)
    RICR_ = scaler.fit_transform(RICR_.T)
    RICR_ = RICR_.T
    RICR_ = RICR_.reshape(dim1, dim0, dim2)
    RICR_normed = np.transpose(RICR_, (1, 0, 2))
    
    return RICR_normed, scaler

def inv_minmax(RICR_normed, scaler):

    dim0, dim1, dim2 = RICR_normed.shape

    RICR_ = np.transpose(RICR_normed, (1, 0, 2))
    RICR_ = RICR_.reshape(dim1, dim0*dim2)
    RICR_ = scaler.inverse_transform(RICR_.T)
    RICR_ = RICR_.T
    RICR_ = RICR_.reshape(dim1, dim0, dim2)
    RICR = np.transpose(RICR_, (1, 0, 2))
    
    return RICR