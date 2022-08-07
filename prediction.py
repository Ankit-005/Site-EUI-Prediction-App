import numpy as np
from sklearn.preprocessing import LabelEncoder

def label(input_val,feat):
    le=LabelEncoder()
    le.fit(feat)
    value=le.transform(np.array([input_val]))
    return value[0]

def predict(data,model):
    return model.predict(data)