from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_random_forest(X_train, y_train, preprocessor):
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    rf_pipeline.fit(X_train, y_train)

    return rf_pipeline
