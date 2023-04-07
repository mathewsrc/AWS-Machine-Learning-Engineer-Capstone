from sklearn.model_selection import train_test_split
import polars as pl
from imblearn.over_sampling import SMOTENC

def split(df, target, use_smote=False, test_size=0.20, val_size=0.10, random_state=42):
    country_idx = df.find_idx_by_name("country")
    gender_idx = df.find_idx_by_name("gender")
    
    X = df.drop(target).to_pandas()
    y = df.select(target).to_pandas()
    
    if use_smote:
        X, y = SMOTENC(categorical_features=[country_idx, gender_idx],
                           random_state=random_state).fit_resample(X, y)
    
    y = y.to_numpy().ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


