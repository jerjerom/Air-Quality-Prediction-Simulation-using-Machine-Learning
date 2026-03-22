import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.load_data import load_data
from src.train_model import train_baseline_model, train_optimized_model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    targets = ["components.co", "components.no2"] # Predicts both CO and NO2 levels

    for target in targets:
        print("\n==============================")
        print(f"Target Variable: {target}")
        print("==============================")

        X_train, X_test, y_train, y_test = load_data("data/air_quality.csv", target) # Load data for the specific target variable

        baseline_model = train_baseline_model(X_train, y_train)
        mae_b, rmse_b, r2_b = evaluate(baseline_model, X_test, y_test)

        optimized_model, best_params = train_optimized_model(X_train, y_train, X_test, y_test)
        mae_o, rmse_o, r2_o = evaluate(optimized_model, X_test, y_test)

        print("\n===== Default Hyperparameters =====")
        print("MAE :", mae_b)
        print("RMSE:", rmse_b)
        print("R2  :", r2_b)

        # print("\n===== Using Grey Wolf Optimization =====")
        # print("MAE :", mae_o)
        # print("RMSE:", rmse_o)
        # print("R2  :", r2_o)

if __name__ == "__main__":
    main()