import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

def get_feature_names(column_transformer):
    """Generate feature names from column transformer"""
    feature_names = []
    for name, pipe, features in column_transformer.transformers_[:-1]: # Skip 'remainder'
        if name == 'cat':
            cats = pipe.named_steps['encoder'].get_feature_names_out(features)
            feature_names.extend(cats)
        else:
            if isinstance(pipe, OneHotEncoder):
                cats = pipe.get_feature_names_out(features)
                feature_names.extend(cats)
            else:
                feature_names.extend(features)
    return feature_names

def preprocess_and_split_data(df, target_column='stroke'):
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    columns_to_drop = ['id']

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)],
        remainder='drop')

    full_pipeline = Pipeline(steps=[
        ('dropper', ColumnDropper(columns_to_drop)),
        ('preprocessor', preprocessor)])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    full_pipeline.fit(X_train)
    X_train_processed = full_pipeline.transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    feature_names = get_feature_names(full_pipeline.named_steps['preprocessor'])

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Remove NaNs after preprocessing
    X_train_processed_df.dropna(inplace=True)
    y_train = y_train[X_train_processed_df.index]
    X_test_processed_df.dropna(inplace=True)
    y_test = y_test[X_test_processed_df.index]

    return X_train_processed_df, X_test_processed_df, y_train, y_test

