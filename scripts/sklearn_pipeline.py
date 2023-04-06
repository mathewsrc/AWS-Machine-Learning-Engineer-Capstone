from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import os

import argparse
import json
import joblib
import polars as pl
import numpy as np
from io import StringIO


from sagemaker_containers.beta.framework import encoders, worker


# Define the column names
feature_columns_names = ['credit_score',
                         'country',
                         'gender',
                         'age',
                         'tenure',
                         'balance',
                         'products_number',
                         'credit_card',
                         'active_member',
                         'estimated_salary'
                        ]


def sklearn_pipeline(train_df):
    """
    Define and fit a preprocessor pipeline for featurizing input data.
    
    Args:
    - train_df: a pandas dataFrame representing the training data.
    
    Returns:
    - preprocessor: a fitted ColumnTransformer object for featurizing input data.
    """
   
    # Define numeric features and transformer
    numeric_features = (
        train_df.select([pl.col(pl.Int64), pl.col(pl.Float64)])
        .select(pl.all().exclude("churn"))
        .columns
    )
    
    print("numeric_features: ", numeric_features)
    
    # Define a pipeline for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    
    # Define categorical features and transformer
    categorical_features = (
        train_df.select(pl.col(pl.Utf8))
        .select(pl.all().exclude("churn"))
        .columns
    )
    
    print("categorical_features: ", categorical_features)

    # Define a pipeline for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
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
    
    # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing
    df_pandas = train_df.drop("churn").to_pandas()
    
    # Fit the preprocessor to the training data
    preprocessor.fit(df_pandas)
    
    # Return the fitted preprocessor
    return preprocessor


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    # Read training data from s3
    train_df = pl.read_csv(os.path.join(args.train, "train.csv"))
            
    preprocessor = sklearn_pipeline(train_df)
    
    # Save the model
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))
    
    print("saved model!")
    
    
def predict_fn(input_data, model):
    """
    Takes input data and a trained model, applies the transform method to preprocess the input data,
    and returns the transformed features. If the input data contains a 'churn' column, it will be inserted 
    at the beginning of the features matrix.
    
    Args:
    - input_data: a pandas dataframe containing the input data.
    - model: a trained model object with a transform() method.
    
    Returns:
    - features: a numpy array or a sparse matrix containing the preprocessed input data.
    """
    
    # Use the transform method to preprocess the input data
    features = model.transform(input_data)
    
    # If the input data contains a 'churn' column, insert it at the beginning of the features matrix
    if "churn" in input_data:
        return np.insert(features, 0, input_data["churn"], axis=1) 
    else:
        return features

        
def model_fn(model_dir):
    """
    Loads a pre-trained model from the specified model directory using joblib.load() and returns it.
    
    Args:
    - model_dir: a string representing the path to the directory containing the trained model.
    
    Returns:
    - preprocessor: a pre-trained scikit-learn model object.
    """
    
    # Load the pre-trained model from the specified model directory using joblib.load()
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    # Return the loaded preprocessor
    return preprocessor

    

def input_fn(input_data, content_type):
    """
    Parses input data in CSV or JSON format and returns a pandas DataFrame object containing the data.
    
    Args:
    - input_data: a string representing the raw input data.
    - content_type: a string representing the content type of the input data.
    
    Returns:
    - df: a pandas DataFrame object containing the input data.
    
    Raises:
    - RuntimeError: if the content type is not supported by the script.
    """
    
    # Parse the input data based on the content type header
    if content_type == 'text/csv':
        
        # Read the input data as a CSV string using polars
        df = pl.read_csv(StringIO(input_data))
 
        # Check if the input data has labels
        if len(df.columns) == len(feature_columns_names) + 1:
            # If it has labels, add the column name "churn" to the last column of the DataFrame
            df.columns = feature_columns_names + ["churn"]
        elif len(df.columns) == len(feature_columns_names):
            # If it does not have labels, add the column names of the features to the DataFrame
            df.columns = feature_columns_names
  
        # Return the parsed DataFrame
        return df.to_pandas()
    else:
        raise RuntimeError("{} not supported by script!".format(content_type))

        
def output_fn(prediction, accept):
    """
    Formats the prediction output as CSV or JSON.
    
    Args:
    - prediction: the model's prediction output.
    - accept: a string representing the desired output format.
    
    Returns:
    - worker.Response: a response object containing the formatted prediction output.
    
    Raises:
    - RuntimeError: if the specified accept type is not supported.
    """
    
    # If the output format is JSON, format the prediction as JSON
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}
        
        return worker.Response(json.dumps(json_output), mimetype=accept)
    
    # If the output format is CSV, format the prediction as CSV
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))