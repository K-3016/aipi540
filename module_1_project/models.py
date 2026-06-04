import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    tf = None


class MajorityBaseline:
    def __init__(self):
        self.most_common_label = None

    def fit(self, y):
        values, counts = np.unique(y, return_counts=True)
        self.most_common_label = values[np.argmax(counts)]

    def predict(self, X):
        if self.most_common_label is None:
            raise ValueError("Baseline has not been fit yet.")
        return np.full(shape=(len(X),), fill_value=self.most_common_label)


class RandomBaseline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Baseline has not been fit yet.")
        rng = np.random.RandomState(self.random_state)
        return rng.choice(self.classes_, size=len(X))


def train_classical_model(X_train, y_train, model_type: str = "random_forest"):
    model_type = model_type.lower()
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, solver="liblinear")
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def build_cnn_model(input_shape, num_classes):
    if tf is None:
        raise ImportError("TensorFlow is required for the deep learning model.")

    model = models.Sequential([
        layers.Rescaling(1.0 / 255.0, input_shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(1 if num_classes == 2 else num_classes, activation="sigmoid" if num_classes == 2 else "softmax"),
    ])
    loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def evaluate_classification(model, X, y):
    y_pred = model.predict(X)
    if hasattr(y_pred, "shape") and y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    report = classification_report(y, y_pred, zero_division=0)
    matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_baseline(baseline, X, y):
    y_pred = baseline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return {"accuracy": accuracy}
