from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, Binarizer

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import polars as pl
import numpy as np



def run_pipeline(dataset):
    """
    This function creates a pipeline that pre-processes and trains a logistic regression classifier on a dataset.
    It includes a numeric transformer for imputing missing values and scaling numeric features, as well as a
    categorical transformer for encoding categorical features. The transformers are combined using a
    ColumnTransformer, and then fed into a logistic regression classifier. The function returns the trained
    classifier.
    """

    df = dataset

    # Define numeric features and transformer
    numeric_features = (
        df.select([pl.col(pl.Int64), pl.col(pl.Float64)])
        .select(pl.all().exclude("customer_id", "churn"))
        .columns
    )
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    
    # Define categorical features and transformer
    categorical_features = (
        df.select(pl.col(pl.Utf8))
        .select(pl.all().exclude("customer_id", "churn"))
        .columns
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehotencoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    preprocessor.fit(df.select(pl.all().exclude("customer_id", "churn")))
    return preprocessor


if __name__ == '__main__':
    LOCAL_ENVIRONMENT = False
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    
    if LOCAL_ENVIRONMENT == True:
        dataset = pl.read_csv("../datasets/bank_customer_churn.csv")
    else:
        dataset = pl.read_csv(args.train + "/bank_customer_churn.csv")
        
    preprocessor = run_pipeline(dataset)
    
    # Save the model
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))
    
    print("saved model!")
    
    
    def predict_fn(input_data, model):
        """ Preprocess input data and make predictions.
        Modify the predict_fn to use .transform() instead of .predict()"""
        
        features = model.transform(input_data)
        
        if "churn" in input_data:
            return np.insert(features, 0, input_data["churn"], axis=1)
        else:
            return features
        
    def model_fn(model_dir):
        """ Deserialize fitted model from model_dir."""
        preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
        return preprocessor