import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, target_column):
    df = pd.read_csv(path)

    manila_df = df[df["city_name"] == "Manila"].copy()
    if len(manila_df) > 1:
        df = manila_df

    df = df.drop(columns=["datetime", "city_name"], errors="ignore")

    # The model uses PM2.5, PM 10, SO2, and O3.
    y = df[target_column]
    X = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test