import pandas as pd 
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

CAT_FEATURES = [ "manufacturer", "model",  "condition", "cylinders","fuel","title_status","transmission", "drive", "size", "type", "paint_color"]
NUM_FEATURES = ["odometer", "year"]


def prepare_df():
    df = pd.read_csv("sources/vehicles.csv", nrows=10000,index_col=0)
    df = df.dropna(subset=["model", "odometer", "year","price"]).copy()
    df= df[df['price'] > 100]

    for cat in CAT_FEATURES:
        df[cat] = df[cat].str.strip().str.replace(' ', '_').str.replace('-', '_').str.lower()
    return df

def create_preprocessor():
    #Numeric
    num_pipe = Pipeline([
        ('scaler',StandardScaler())
    ])

    #Categocial 
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(
                        min_frequency=10, 
                        handle_unknown='infrequent_if_exist',
                        sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, NUM_FEATURES),
        ('cat', cat_pipe, CAT_FEATURES)])
    
    return preprocessor

def create_pipeline(preprocessor):
    xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
    ], memory='./cache_dir')

    param_distributions = {
    'model__eta': np.arange(0,1,0.15),
    'model__max_depth': [None, 10, 20, 30, 40, 50],
    'model__min_child_weight' : [0, 1, 5, 10, 20, 30, 50],
    'model__subsample' : [0.25, 0.5, 0.75, 1],
    'model__lambda' : [0, 1, 3, 5, 10]
    }

    random_search = RandomizedSearchCV(
    xgb_pipeline, 
    param_distributions=param_distributions, 
    n_iter=50, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1, 
    random_state=42 
    )
    return random_search


def train_pipeline(df, rs_pipeline):
    y = np.log1p(df["price"])                            
    X = df[CAT_FEATURES + NUM_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rs_pipeline.fit(X_train, y_train)
    return rs_pipeline 

def save_model(rs_pipeline):
    best_pipeline = rs_pipeline.best_estimator_
    joblib.dump(best_pipeline, 'model.pkl')
