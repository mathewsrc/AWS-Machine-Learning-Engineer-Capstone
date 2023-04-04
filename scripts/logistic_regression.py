from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import polars as pl
import argparse
import os
import pandas as pd

# first column is the target variable
columns_name = [ 'churn',
                 'credit_score',
                 'country.A',
                 'country.B',
                 'country.C',
                 'gender.A',
                 'gender.B',
                 'age',
                 'tenure',
                 'balance',
                 'products_number',
                 'credit_card',
                 'active_member',
                 'estimated_salary' ]


def train(X_train, y_train, args):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
     # Save the model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print(f"Model saved at: {path}")
    return model

def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    lr_auc = auc(fpr, tpr)
    
    print("Accuracy: {:.3f}".format(acc))
    print("Classification Report:\n", report)
    print("AUC: {:.3f}".format(lr_auc))
             
def model_fn(model_dir):
        """ Deserialize fitted model from model_dir."""
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        return model
    
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str,default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args = parser.parse_args()
    
    #train_df = pl.read_csv(os.path.join(args.train, "train.csv.out"), has_header=False)
    #test_df = pl.read_csv(os.path.join(args.test, "test.csv.out"), has_header=False)
    
    train_df = pd.read_csv(os.path.join(args.train, "train.csv.out"), header=None)
    test_df = pd.read_csv(os.path.join(args.test, "test.csv.out"), header=None)

    train_df.columns = columns_name
    test_df.columns = columns_name
    
    train_df = pl.from_pandas(train_df)
    test_df = pl.from_pandas(test_df)
    
    X_train = train_df.select(pl.all().exclude("churn")).to_pandas()
    y_train =  train_df.select("churn").to_pandas()
    
    X_test = test_df.select(pl.all().exclude("churn")).to_pandas()
    y_test =  test_df.select("churn").to_pandas()
    
    model = train(X_train, y_train, args)
    test(model, X_test, y_test)
