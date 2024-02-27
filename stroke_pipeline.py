import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def train_test_split_stroke(df):
    X = df.drop(columns='y')
    y = df['y'].values.reshape(-1,1)
    return train_test_split(X, y, random_state=13)


# Functions for filling values
def fill_gender(X):
    X['gender'] = X['gender'].fillna('unknown')
    return X

def fill_marriage(X):
    X['ever_married'] = X['ever_married'].fillna('primary')
    return X

def fill_work_type(X):
    X['work_type'] = X['work_type'].fillna('unknown')
    return X

def fill_residence_type(X):
    X['Residence_type'] = X['Residence_type'].fillna('unkown')
    return X

def fill_smoking(X):
    X['smoking_status'] = X['smoking_status'].fillna('unknown')
    return X

def fill_bmi(X):
    X['BMI'] = X['BMI'].fillna('unkown')
    return X

def fill_missing(X):
    X = fill_gender(X)
    X = fill_marriage(X)
    X = fill_work_type(X)
    X = fill_residence_type(X)
    X = fill_smoking(X)
    X = fill_bmi(X)
    return X


# Functions for building and training encoders
def build_gender_encoder(X_filled):
    gender_encoder = OneHotEncoder(max_categories=2, handle_unknown='ignore', sparse_output=False)
    # Train the encoder
    gender_encoder.fit(X_filled['gender'].values.reshape(-1, 1))
    return {'column': 'gender',
            'multi_col_output': True,
            'encoder': gender_encoder}

def build_marriage_encoder(X_filled):
    marriage_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    marriage_encoder.fit(X_filled['ever_married'].values.reshape(-1, 1))
    return {'column': 'ever_married',
            'multi_col_output': True,
            'encoder': marriage_encoder}

def build_work_type_encoder(X_filled):
    work_type_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    work_type_encoder.fit(X_filled['work_type'].values.reshape(-1, 1))
    return {'column': 'work_type',
            'multi_col_output': True,
            'encoder': work_type_encoder}

def build_residence_type_encoder(X_filled):
    residence_type_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    residence_type_encoder.fit(X_filled['Residence_type'].values.reshape(-1, 1))
    return {'column': 'Residence_type',
            'multi_col_output': True,
            'encoder': residence_type_encoder}

def build_smoking_encoder(X_filled):
    smoking_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    smoking_encoder.fit(X_filled['smoking_status'].values.reshape(-1, 1))
    return {'column': 'smoking_status',
            'multi_col_output': True,
            'encoder': smoking_encoder}


def build_encoders(X_filled):
    encoder_functions = [build_gender_encoder, 
                         build_marriage_encoder, 
                         build_work_type_encoder, 
                         build_residence_type_encoder,
                         build_smoking_encoder
                        ]
    return [encoder_function(X_filled) for encoder_function in encoder_functions]


# Encoding all categorical variables
def encode_categorical(X_filled, encoders):
    # Separate numeric columns
    dfs = [X_filled.select_dtypes(include='number').reset_index(drop=True)]

    single_col_encoders = []
    for encoder_dict in encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        if not multi_col:
            single_col_encoders.append(encoder_dict)
        else:
            dfs.append(pd.DataFrame(encoder.transform(X_filled[column].values.reshape(-1, 1)), columns=encoder.get_feature_names_out()))
    
    X_encoded = pd.concat(dfs, axis=1)

    for encoder_dict in single_col_encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        X_encoded[column] = encoder.transform(X_filled[column].values.reshape(-1, 1))

    return X_encoded

def build_target_encoder(y):
    encode_y = OneHotEncoder(drop='first', sparse_output=False)
    encode_y.fit(y)
    return encode_y

def encode_target(y, encode_y):
    
    return np.ravel(encode_y.transform(y))
