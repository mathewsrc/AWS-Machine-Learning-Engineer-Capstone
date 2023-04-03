from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import polars as pl
import argparse
import os



def train_logistic_regression(X_train, y_train):
    clf_model = LogisticRegression(random_state=42)
    # Train classifier using training data
    clf_model.fit(X_train, y_train)

    return  clf_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    dataset = pl.read_csv(os.path.join(args.train, "train.csv.out"))
    
    X_train = dataset.to_pandas().iloc[:, :-1] # First 10 columns
    y_train = dataset.to_pandas().iloc[:, -1]  # Last column
        
    model = train_logistic_regression(X_train, y_train)
    
    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
    print("saved model!")
