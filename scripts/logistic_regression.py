from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def train_logistic_regression(X_train, y_train, preprocessor):
    # Define final pipeline including preprocessor and logistic regression classifier
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Train classifier using training data
    clf_pipeline.fit(X_train, y_train)

    return clf_pipeline
