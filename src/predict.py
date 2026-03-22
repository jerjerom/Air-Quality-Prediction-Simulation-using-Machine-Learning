import joblib


def predict(model_path, X_test):
    model = joblib.load(model_path)
    return model.predict(X_test)