import json
import os
import pickle
import numpy as np
__locations = None
__data_columns =  None
__model = None


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Get the path relative to the current script file
    base_dir = os.path.dirname(__file__)
    artifacts_path = os.path.join(base_dir, "artifacts")

    with open(os.path.join(artifacts_path, "columns.json"), 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open(os.path.join(artifacts_path, "Banglore_House_Data_model.pickle"), 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")


if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_names())