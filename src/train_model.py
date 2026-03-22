import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_baseline_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/random_forest_baseline.pkl")
    return model


def fitness(params, X_train, y_train, X_test, y_test):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    max_features = float(params[2])

    # Random Forest with the default hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)


# Grey Wolf Optimization for hyperparameter tuning
def grey_wolf_optimization(X_train, y_train, X_test, y_test, wolves=10, iterations=20):
    bounds = np.array(
        [
            [50, 300],  # n_estimators
            [3, 20],  # max_depth
            [0.3, 1.0],  # max_features
        ]
    )

    dim = bounds.shape[0]

    positions = np.random.rand(wolves, dim)
    positions = bounds[:, 0] + positions * (bounds[:, 1] - bounds[:, 0])

    alpha = None
    beta = None
    delta = None

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    for iteration in range(iterations):
        for i in range(wolves):
            score = fitness(positions[i], X_train, y_train, X_test, y_test)

            if score < alpha_score:
                delta_score = beta_score
                delta = None if beta is None else beta.copy()

                beta_score = alpha_score
                beta = None if alpha is None else alpha.copy()

                alpha_score = score
                alpha = positions[i].copy()

            elif score < beta_score:
                delta_score = beta_score
                delta = None if beta is None else beta.copy()

                beta_score = score
                beta = positions[i].copy()

            elif score < delta_score:
                delta_score = score
                delta = positions[i].copy()

        a = 2 - iteration * (2 / iterations)

        for i in range(wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - positions[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - positions[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - positions[i][j])
                X3 = delta[j] - A3 * D_delta

                positions[i][j] = (X1 + X2 + X3) / 3
                positions[i][j] = np.clip(positions[i][j], bounds[j][0], bounds[j][1])

    return alpha


def train_optimized_model(X_train, y_train, X_test, y_test):
    best_params = grey_wolf_optimization(X_train, y_train, X_test, y_test)

    model = RandomForestRegressor(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        max_features=float(best_params[2]),
        random_state=42,
    )

    model.fit(X_train, y_train)
    joblib.dump(model, "models/random_forest_optimized.pkl")

    return model, best_params
