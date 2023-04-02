from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def train_decision_tree(X_train, y_train, preprocessor):

    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    # Train classifier using training data
    clf_pipeline.fit(X_train, y_train)

    return clf_pipeline
