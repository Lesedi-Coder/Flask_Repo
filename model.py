"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
riders = pd.read_csv('utils/data/riders.csv')

def find_true_distance(Pick_up_lat, Pick_up_long, Destination_lat, Destination_long):
    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    Pick_up_lat = Pick_up_lat.apply(lambda x: radians(x))
    Pick_up_long = Pick_up_long.apply(lambda x: radians(x))
    Destination_lat = Destination_lat.apply(lambda x: radians(x))
    Destination_long = Destination_long.apply(lambda x: radians(x))

    dlon = Destination_long - Pick_up_long
    dlat = Destination_lat - Pick_up_lat 

    a = dlat.apply(lambda x: sin( x/ 2)**2) + Pick_up_lat.apply(lambda x: cos(x)) * Destination_lat.apply(lambda x: cos(x)) * dlon.apply(lambda x : sin(x / 2)**2)
    c = 2 * a.apply(lambda x: atan2(sqrt(x), sqrt(1 - x)))

    return round(R * c)

def convert_to_24hrs(column):
    column = pd.to_datetime(column, format='%I:%M:%S %p').dt.strftime("%H:%M:%S")
    return column

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    train_set = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    train_set.drop(['Arrival at Destination - Day of Month', 'Arrival at Destination - Weekday (Mo = 1)', 'Arrival at Destination - Time'], axis=1, inplace=True)

    train_set = pd.merge(train_set, riders, on='Rider Id', how= 'left')
    column_titles = [col for col in train_set.columns if col!= 'Time from Pickup to Arrival'] + ['Time from Pickup to Arrival']
    train_set= train_set.reindex(columns=column_titles)
    
    train_set['Temperature'] = train_set['Temperature'].fillna(train_set['Temperature'].mean())
    train_set['Precipitation in millimeters'] = train_set['Precipitation in millimeters'].fillna(0)

    train_set.drop(['User Id', 'Vehicle Type','Order No', 'Rider Id' ],axis=1, inplace=True)

    train_set['Confirmation - Time'] = pd.to_timedelta(convert_to_24hrs(train_set['Confirmation - Time'] ))
    train_set['Placement - Time'] = pd.to_timedelta(convert_to_24hrs(train_set['Placement - Time'] ))
    train_set['Arrival at Pickup - Time'] =  pd.to_timedelta(convert_to_24hrs(train_set['Arrival at Pickup - Time'] ))
    train_set['Pickup - Time'] =  pd.to_timedelta(convert_to_24hrs(train_set['Pickup - Time'] ))

    train_set['Time from Confirmation to placement'] = train_set['Confirmation - Time']- train_set['Placement - Time']
    train_set['Waiting Time'] = train_set['Pickup - Time']- train_set['Arrival at Pickup - Time']

    train_set['Time from Confirmation to placement'] = train_set['Time from Confirmation to placement'] / np.timedelta64(1, 's')
    train_set['Waiting Time'] = train_set['Waiting Time'] / np.timedelta64(1, 's')

    train_set.drop(['Confirmation - Time','Placement - Time' ,'Pickup - Time','Arrival at Pickup - Time'],
             axis=1, inplace=True)

    column_titles = [col for col in train_set.columns if col!= 'Time from Pickup to Arrival'] + ['Time from Pickup to Arrival']
    train_set= train_set.reindex(columns=column_titles)

    train_set.drop([ 'Placement - Day of Month', 'Placement - Weekday (Mo = 1)', 'Arrival at Pickup - Day of Month', 'Arrival at Pickup - Weekday (Mo = 1)', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'No_of_Ratings', 'Age'],
             axis=1, inplace=True)

    train_set['True Distance'] = find_true_distance(train_set['Pickup Lat'], train_set['Pickup Long'], train_set['Destination Lat'], train_set['Destination Long'])

    train_set.drop(['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long'],
             axis=1, inplace=True)

    column_titles = [col for col in train_set.columns if col!= 'Time from Pickup to Arrival'] + ['Time from Pickup to Arrival']
    train_set= train_set.reindex(columns=column_titles)        

    #y_train = train_set.iloc[:,-1].values
    #y_train.reshape(len(y_train),1)

    X_train = train_set.iloc[:,:-1].values
  
    ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(drop='first'), [0,1])], remainder = 'passthrough')
    X_train = np.array(ct.fit_transform(X_train))

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)

    # ------------------------------------------------------------------------

    return X_train

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
